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

import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from torchmetrics import JaccardIndex

import gc
from pathlib import Path
import shutil


def upsample(masks, target_side=800):
    in_w = masks.shape[-1]
    in_l = masks.shape[-2]
    min_side = min(in_w, in_l)
    scale_factor = target_side / min_side
    target_size = (int(in_l * scale_factor), int(in_w * scale_factor))
    upsampled = F.interpolate(masks.float().unsqueeze(1), size=target_size, mode='nearest').squeeze(1).bool()
    return upsampled, in_w, in_l


def downsample(masks, original_w, original_l):
    target_size = (original_l, original_w)
    downsampled = F.interpolate(masks.float().unsqueeze(1), size=target_size, mode='nearest').squeeze(1).bool()
    return downsampled


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

        # Enable the saving of segmentation images.
        # TODO !!! MAKE SURE THE IMAGE SCALING IS SET TO [1, 1], WHEN RUNNING FOR VISUALIZATIONS !!!
        save_visualizations = False  # MAKE SURE YOU SET THE OUTPUT FOLDER CORRECTLY, WHEN YOU ENABLE THIS.
        root_output_folder = '/workspace/data/EOMT/Output/'
        output_folder = root_output_folder + 'val_run2_on_202408_val/'
        if batch_idx == 0 and save_visualizations:
            # Make sure output folder exists and is empty.
            folder_path = Path(output_folder)
            if folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(exist_ok=True, parents=True)

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

            # Upsample the masks. We believe this forces the code that computes the metrics to encode the masks efficiently.
            preds[0]['masks'], in_w, in_l = upsample(preds[0]['masks'])
            targets_[0]['masks'], _, _,  = upsample(targets_[0]['masks'])
            self.update_metrics_instance(preds, targets_, i)  # The original code did not have the underscore, but it seems that COCO uses the "iscrowd" version of the filed.

        if save_visualizations:
            # Downsample the predictions and the targets to the original image size.
            p_masks = downsample(preds[0]['masks'], in_w, in_l)
            t_masks = downsample(targets_[0]['masks'], in_w, in_l)
            self.visualize_instance_segmentation(preds[0],
                                                 p_masks,
                                                 imgs[0],
                                                 batch_idx,
                                                 t_masks,
                                                 output_folder=output_folder,  # MAKE SURE YOU SET THIS CORRECTLY.
                                                 image_path = targets[0].get('image_path', None),
                                                 score_thresh=0.8)  # TODO: What value would be good?
        # Release memory.
        del imgs, targets, transformed_imgs, mask_logits_per_layer, class_logits_per_layer, mask_logits, preds, targets_, scores, labels, topk_indices, topk_scores, masks, mask_scores
        if save_visualizations:
            del p_masks, t_masks
        gc.collect()
        torch.cuda.empty_cache()

    def visualize_instance_segmentation(self, preds, p_masks, image, batch_idx, t_masks, output_folder, image_path, score_thresh=0.8):
        '''Saves images with the segmentation (and the GT) to a folder, as well as the iou_log.csv file.'''
        with torch.no_grad():
            metric = JaccardIndex(task='binary')


            # Variable names
            #################
            # overlaid_detection_image: input image with blended segment overlays.
            # crisp_detection_image: image with crisp segments, used for IoU computation.
            # overlaid_gt_image: GT image with blended segment overlays.
            # crisp_gt_image: GT image with crisp segments, used for IoU computation.


            ##################################################
            # COMPUTE THE COLOR IMAGE WITH SEGMENT OVERLAYS, #
            # AND THE IMAGE WITH CRISP DETECTION SEGMENTS.   #
            ##################################################
            # Segments are overlaid on the color image with alpha blending.
            alpha_color_image = 0.4  # blending factor, 0.0 = no overlay, 1.0 = full overlay.

            masks = p_masks
            labels = preds["labels"]
            scores = preds["scores"]

            # Filter by confidence threshold
            keep = scores > score_thresh
            masks, labels, scores = masks[keep], labels[keep], scores[keep]

            # This is a copy of the input image, on which segments will be overlaid.
            if isinstance(image, torch.Tensor):
                overlaid_detection_image = image.permute(1, 2, 0).cpu().numpy()
            else:
                overlaid_detection_image = np.array(image)

            # This image is used to compute the IoU with the GT.
            crisp_detection_image = np.zeros(overlaid_detection_image.shape)

            if len(masks) > 0:
                n_detections = masks.shape[0]
                # Create instance map
                inst_map = torch.zeros_like(masks[0], dtype=torch.int32)
                for i, mask in enumerate(masks):
                    inst_map[mask > 0.5] = i + 1  # assign unique ID

                # Generate random colors for instances
                num_instances = len(masks)
                colors = list(np.random.permutation(generate_bgr_colors(num_instances)))

                for i in range(num_instances):
                    mask = masks[i].cpu().numpy() > 0.5
                    color = np.array(colors[i % len(colors)])
                    # Alpha blend only where mask is True.
                    overlaid_detection_image[mask] = (1 - alpha_color_image) * overlaid_detection_image[mask] + alpha_color_image * color
                    # Add the segment to the crisp image.
                    crisp_detection_image[mask] = [0.0, 1.0, 0.0]
            else:
                crisp_detection_image = overlaid_detection_image
                n_detections = 0
                colors = list(np.random.permutation(generate_bgr_colors(1)))

            #################################################
            # COMPUTE THE GT IMAGE, WITH BLENDED OVERLAYS,  #
            # AND THE IMAGE WITH CRISP GT SEGMENTS.         #
            #################################################
            # These start out as black images.
            crisp_gt_image = np.zeros(overlaid_detection_image.shape)
            overlaid_gt_image = np.zeros(overlaid_detection_image.shape)
            alpha_gt_image = 0.7  # blending factor, 0.0 = no overlay, 1.0 = full overlay.


            gt_masks = t_masks  # targets['masks']
            if gt_masks.dim() == 3:    # sometimes it's [1, H, W]
                gt_masks = gt_masks.squeeze(0)
            gt_masks = gt_masks.cpu().numpy()
            try:
                if len(gt_masks.shape) == 2:
                    n_gt_masks = 1
                    colors = list(np.random.permutation(generate_bgr_colors(1)))
                    crisp_gt_image[gt_masks] = [0.0, 1.0, 0.0]
                    color = np.array(colors[0])
                    overlaid_gt_image[gt_masks] = (1 - alpha_gt_image) * overlaid_gt_image[gt_masks] + alpha_gt_image * color
                else:
                    n_gt_masks = gt_masks.shape[0]
                    colors = list(np.random.permutation(generate_bgr_colors(n_gt_masks)))
                    for index, current_mask in enumerate(gt_masks):
                        crisp_gt_image[current_mask] = [0, 1, 0]
                        # Alpha blend only where mask is True.
                        color = np.array(colors[index % len(colors)])
                        overlaid_gt_image[current_mask] = (1 - alpha_gt_image) * overlaid_gt_image[current_mask] + alpha_gt_image * color

            except Exception:
                print(f'ERROR: GT image could not be computed for batch {batch_idx}!')
                crisp_gt_image = np.zeros(overlaid_detection_image.shape)
                n_gt_masks = 0

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(overlaid_detection_image)
            axes[0].set_title("Detections")
            axes[1].imshow(overlaid_gt_image/255.0)
            axes[1].set_title("GT")

            if len(masks) > 0:
                iou = metric(
                    torch.asarray(crisp_detection_image[:,:,1]),
                    torch.asarray(crisp_gt_image[:,:,1]))
            else:
                iou = torch.tensor(0.0)
            # print(f'BatchIndex {batch_idx} IoU {iou.item()}')

            with open(f'{output_folder}/iou_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{batch_idx:05d}", image_path, f"{iou.item():0.2f}", n_detections, n_gt_masks, n_detections - n_gt_masks])
            fig.suptitle(f"SemSeg IoU = {iou.item():0.2f}", fontsize=24)

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{batch_idx:05d}.jpg")
            plt.close()
            plt.close(fig)
            del fig
            torch.cuda.empty_cache()
            del overlaid_detection_image, crisp_detection_image, labels, scores, gt_masks, crisp_gt_image
            if len(masks) > 0:
                del inst_map, masks

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")
