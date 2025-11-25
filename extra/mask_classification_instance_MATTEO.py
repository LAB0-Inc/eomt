# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule

# Matteo's imports.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from torchmetrics import JaccardIndex


def generate_bgr_colors(n):
    """
    Generate n fully saturated, equally spaced colors along the hue circle in BGR.

    Args:
        n (int): Number of colors.

    Returns:
        List of n colors in BGR format (values 0-255).
    """
    # Equally spaced hues [0, 180) for OpenCV's HSV (H ranges 0-179)
    hues = np.linspace(0, 179, n, endpoint=False).astype(int)

    # Full saturation and value
    s = np.full_like(hues, 255)
    v = np.full_like(hues, 255)

    # Stack into HSV array
    hsv_colors = np.stack([hues, s, v], axis=1).astype(np.uint8)

    # Convert to BGR using OpenCV
    bgr_colors = cv2.cvtColor(hsv_colors[np.newaxis, :, :], cv2.COLOR_HSV2BGR)[0]

    # Return as a list of tuples
    return [tuple(int(c) for c in color) for color in bgr_colors]


class MaskClassificationInstance(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes: List[int] = []
        self.eval_top_k_instances = eval_top_k_instances

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_instance(self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    self.eval_top_k_instances, sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                mask_logits[j] = mask_logits[j][topk_indices]

                masks = mask_logits[j] > 0
                mask_scores = (
                    mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                preds.append(
                    dict(
                        masks=masks,
                        labels=labels,
                        scores=scores,
                    )
                )
                targets_.append(
                    dict(
                        masks=targets[j]["masks"],
                        labels=targets[j]["labels"],
                        iscrowd=targets[j]["is_crowd"],
                    )
                )

            self.update_metrics_instance(preds, targets, i)

        # MATTEO.
        self.visualize_instance_segmentation(preds[0], imgs[0], batch_idx, targets[0], score_thresh=0.8)

    def visualize_instance_segmentation(self, preds, image, batch_idx, targets, score_thresh=0.8):
        '''MATTEO TEMP'''

        #################################################
        # COMPUTE THE COLOR IMAGE WITH SEGMENT OVERLAYS #
        #################################################
        masks = preds["masks"].squeeze(1)  # shape [N, H, W]
        labels = preds["labels"]
        scores = preds["scores"]

        # Filter by confidence threshold
        keep = scores > score_thresh
        masks, labels, scores = masks[keep], labels[keep], scores[keep]

        # Create instance map
        inst_map = torch.zeros_like(masks[0], dtype=torch.int32)
        for i, mask in enumerate(masks):
            inst_map[mask > 0.5] = i + 1  # assign unique ID

        # Generate random colors for instances
        num_instances = len(masks)
        colors = list(np.random.permutation(generate_bgr_colors(num_instances)))

        # Create a color visualization
        if isinstance(image, torch.Tensor):
            color_img = image.permute(1, 2, 0).cpu().numpy()
        else:
            color_img = np.array(image)

        # Alpha blending
        alpha = 0.5  # blending factor, 0.0 = no overlay, 1.0 = full overlay

        combined_det_image = np.zeros(color_img.shape)
        for i in range(num_instances):
            mask = masks[i].cpu().numpy() > 0.5
            color = np.array(colors[i])
            # Alpha blend only where mask is True
            color_img[mask] = (1 - alpha) * color_img[mask] + alpha * color
            combined_det_image[mask] = [0.0, 1.0, 0.0]

        #################################################
        # COMPUTE THE GT IMAGE, BLACK WITH BOX SEGMENTS #
        #################################################
        gt_image = np.zeros(color_img.shape)
        gt_masks = targets['masks']
        if gt_masks.dim() == 3:    # sometimes it's [1, H, W]
            gt_masks = gt_masks.squeeze(0)
        gt_masks = gt_masks.cpu().numpy()
        try:
            for index, current_mask in enumerate(gt_masks):
                gt_image[current_mask] = [0.0, 1.0, 0.0]
        except Exception:
            print('ERROR: GT image could not be computed!')
            gt_image = np.zeros(color_img.shape)

        # plt.figure(figsize=(8, 8))
        # plt.imshow(color_img)
        # plt.axis("off")
        # plt.savefig(f"/workspace/data/EOMT/Output/VAL_STEP/{batch_idx:04d}.png")
        # plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(color_img)
        # axes[0].imshow(combined_det_image)
        axes[0].set_title("Detections")
        axes[1].imshow(gt_image)
        axes[1].set_title("GT")

        metric = JaccardIndex(task='binary')
        iou = metric(torch.tensor(combined_det_image[:,:,1]), torch.tensor(gt_image[:,:,1]))
        # print(f'BatchIndex {batch_idx} IoU {iou.item()}')
        with open("/workspace/data/EOMT/Output/TRAIN_STEP/iou_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([batch_idx, iou.item()])
        fig.suptitle(f"SemSeg IoU = {iou.item()}", fontsize=24)

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"/workspace/data/EOMT/Output/TRAIN_STEP/{batch_idx:04d}.png")
        plt.close()

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")
