import yaml
from lightning import seed_everything
import torch
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import numpy as np
import warnings
import importlib
import os
import json
from pycocotools import mask as mask_utils


def infer_panoptic(img, target, mask_thresh, overlap_thresh):
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):

        # 1. Scale input.
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        # print("Original shape:", imgs[0].shape)
        # print("Transformed shape:", transformed_imgs.shape)
        del img, imgs  # Free memory.

        # 2. Apply network.
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)

        # 3. Go back to original image size.
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )

        # 4. Work on detections, transform them.
        preds_gpu, inst_scores = local_to_per_pixel_preds_panoptic(
            mask_logits,
            class_logits_per_layer[-1],
            model.stuff_classes,
            mask_thresh,     # model.mask_thresh,     There is a value stored in the model.
            overlap_thresh,  # model.overlap_thresh,  There is a value stored in the model.
            model.num_classes  # This is 2, should it not be 1?
        )
        preds = preds_gpu[0].cpu()

    pred = preds.numpy()

    # inst_pred is the matrix that encodes the instance segmentation segments. Each segments has a different ID. -1 means background.
    sem_pred, inst_pred = pred[..., 0], pred[..., 1]

    target_seg = model.to_per_pixel_targets_panoptic([target])[0].cpu().numpy()
    sem_target, inst_target = target_seg[..., 0], target_seg[..., 1]

    return sem_pred, inst_pred, sem_target, inst_target, inst_scores


def local_to_per_pixel_preds_panoptic(mask_logits_list, class_logits, stuff_classes, mask_thresh, overlap_thresh, num_classes
):
    scores, classes = class_logits.softmax(dim=-1).max(-1)
    preds_list = []
    scores_list = []

    for i in range(len(mask_logits_list)):
        preds = -torch.ones(
            (*mask_logits_list[i].shape[-2:], 2),
            dtype=torch.long,
            device=class_logits.device,
        )
        preds[:, :, 0] = num_classes

        keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
        if not keep.any():
            preds_list.append(preds)
            scores_list.append([])
            continue

        masks = mask_logits_list[i].sigmoid()
        segments = -torch.ones(
            *masks.shape[-2:],
            dtype=torch.long,
            device=class_logits.device,
        )

        mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
        stuff_segment_ids, segment_id = {}, 0
        segment_and_class_ids = []
        segment_scores = []

        kept_scores = scores[i][keep]
        for k, class_id in enumerate(classes[i][keep].tolist()):
            orig_mask = masks[keep][k] >= 0.5
            new_mask = mask_ids == k
            final_mask = orig_mask & new_mask

            orig_area = orig_mask.sum().item()
            new_area = new_mask.sum().item()
            final_area = final_mask.sum().item()
            if (
                orig_area == 0
                or new_area == 0
                or final_area == 0
                or new_area / orig_area < overlap_thresh
            ):
                continue

            if class_id in stuff_classes:
                if class_id in stuff_segment_ids:
                    segments[final_mask] = stuff_segment_ids[class_id]
                    continue
                else:
                    stuff_segment_ids[class_id] = segment_id

            segments[final_mask] = segment_id
            segment_and_class_ids.append((segment_id, class_id))
            segment_scores.append(kept_scores[k].item())

            segment_id += 1

        for segment_id, class_id in segment_and_class_ids:
            segment_mask = segments == segment_id
            preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
            preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

        preds_list.append(preds)
        scores_list.append(segment_scores)

    return preds_list, scores_list[0]


def draw_black_border(sem, inst, color):
    h, w = sem.shape
    out = np.zeros((h, w, 3))
    for s in np.unique(sem):
        out[sem == 1] = color

    combined = sem.astype(np.int64) * 100000 + inst.astype(np.int64)
    border = np.zeros((h, w), dtype=bool)
    border[1:, :] |= combined[1:, :] != combined[:-1, :]
    border[:-1, :] |= combined[1:, :] != combined[:-1, :]
    border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
    border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
    out[border] = 0
    return out


def plot_panoptic_results(img, sem_pred, inst_pred, sem_target, inst_target, output_path):
    all_ids = np.union1d(np.unique(sem_pred), np.unique(sem_target))
    mapping = {
        s: (
            [0, 0, 0]
            if s == -1 or s == model.num_classes
            else plt.cm.hsv(i / len(all_ids))[:3]
        )
        for i, s in enumerate(all_ids)
    }

    vis_pred = draw_black_border(sem_pred, inst_pred, [1, 0, 0])
    vis_target = draw_black_border(sem_target, inst_target, [0, 1, 0])
    vis_pred = vis_pred + vis_target
    img_np = (
        img.cpu().numpy().transpose(1, 2, 0) if img.dim() == 3 else img.cpu().numpy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[1].imshow(vis_pred)
    axes[1].set_title("Prediction (red) + target (green)")
    axes[2].imshow(vis_target)
    axes[2].set_title("Target")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_path}.jpg")
    plt.close()


def instance_segmentation_to_coco(inst_pred, inst_scores, class_id, image_id, category_id, annotation_id_start=1):
    """
    Convert instance segmentation prediction to COCO format annotations.

    Args:
        inst_pred: numpy array (H, W) with instance IDs (-1 for background)
        inst_scores: list of confidence scores for each instance
        class_id: the semantic class ID for these instances
        image_id: COCO image ID
        category_id: COCO category ID for this class
        annotation_id_start: starting annotation ID

    Returns:
        list of COCO annotation dictionaries
    """
    annotations = []
    annotation_id = annotation_id_start

    # Get unique instance IDs (excluding -1 for background)
    instance_ids = np.unique(inst_pred)
    instance_ids = instance_ids[instance_ids >= 0]

    for inst_id in instance_ids:
        # Extract binary mask for this instance
        binary_mask = (inst_pred == inst_id).astype(np.uint8)

        # Convert to COCO RLE format (Fortran order)
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON

        # Calculate bounding box [x, y, width, height]
        rows, cols = np.where(binary_mask)
        if len(rows) == 0:
            continue

        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()
        bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

        # Calculate area
        area = int(binary_mask.sum())

        # Get confidence score
        score = inst_scores[inst_id] if inst_id < len(inst_scores) else 1.0

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "score": float(score)
        }

        annotations.append(annotation)
        annotation_id += 1

    return annotations



############
# Settings #
############

# Generic settings.
device = 0  # GPU[0].
seed_everything(0, verbose=False)
# Filter some warnings.
warnings.filterwarnings(
    "ignore", message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
)

# Deep Learning settings.
# config_path = "configs/dinov3/coco/instance/eomt_large_640.yaml"
# data_path = "/workspace/data/Datasets/SCD_for_EoMT/"
# config_path = "configs/dinov3/coco/instance/eomt_large_640_our_202408.yaml"
# data_path = "/workspace/data/Datasets/202408/"
config_path = "configs/dinov3/coco/instance/eomt_large_640_validate_on_ft_training.yaml"
data_path = "/workspace/data/Datasets/fine_tuning_2025_12_16"
masked_attn_enabled = False

# my_checkpoint = '/workspace/data/EOMT/Checkpoints/Run1/epoch=026-mAP=0.81.ckpt'  # Should the mAP not reach 86?
# output_dir = '/workspace/data/EOMT/Output/val_run1/'
#
# my_checkpoint = '/workspace/data/EOMT/Checkpoints/Run2/epoch=055-mAP=0.81.ckpt'  # Should the mAP not reach 86?
# output_dir = '/workspace/data/EOMT/Output/val_run2_mask_attn_off/'
#
# my_checkpoint = '/workspace/data/EOMT/Checkpoints/Run2/epoch=055-mAP=0.81.ckpt'  # Should the mAP not reach 86?
# output_dir = '/workspace/data/EOMT/Output/Temp/'

# Best model after fine turning
my_checkpoint = '/workspace/data/EOMT/Checkpoints/FT1_3/epoch=018-mAP=0.88.ckpt'
output_dir = '/workspace/data/EOMT/Output/val_on_ft_training/'

os.makedirs(output_dir, exist_ok=True)



#####################
# Prepare the model #
#####################
# 1. Load the configuration.
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# 2. Set up the data loader.
data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
data_module = getattr(importlib.import_module(data_module_name), class_name)
data_module_kwargs = config["data"].get("init_args", {})
data = data_module(
    path=data_path,
    batch_size=1,
    num_workers=0,
    check_empty_targets=False,
    **data_module_kwargs
).setup()


# 3. Set up the model, we are basically building the Lightning module by hand.
# 3.1. Load the encoder (it is given as input when creating the network).
encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
encoder = encoder_cls(img_size=data.img_size, **encoder_cfg.get("init_args", {}))

# 3.2. Load the network.
network_cfg = config["model"]["init_args"]["network"]
network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
network = network_cls(
    masked_attn_enabled=masked_attn_enabled,
    num_classes=data.num_classes,
    encoder=encoder,
    **network_kwargs)

# 3.3. Load the Lightning module.
lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
if "stuff_classes" in config["data"].get("init_args", {}):
    model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

# 3.4. Build the model using Lightning, while loading the weights from a checkpoint.
model = lit_cls.load_from_checkpoint(
    my_checkpoint,
    img_size=data.img_size,
    num_classes=data.num_classes,
    network=network,
    ckpt_path=None,  # TODO - WARNING - Disable nested checkpoint loading. WHAT IS IT TRYING TO LOAD?
    **model_kwargs).eval().to(device)



####################################################
# Run instance segmentation on the validation set. #
####################################################
mask_thresh = 0.8  # Detections with a confidence below this value are rejected.
overlap_thresh = 0.8  # Detections that after detection overlap analysis are shrunk to less than this fraction of their area.

val_dataset = data.val_dataloader().dataset
step = 1

predictions = []
coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}
annotation_id = 1

# for img_idx in [0]:                                 # Single image.
for img_idx in range(0, len(val_dataset), step):  # Full validation set.
    # Get the sample.
    img, target = val_dataset[img_idx]
    image_id = target.get('index') + 1  # COCO starts numbering samples from 1, not 0.
    assert(image_id == img_idx + 1)  # They should be the same, we are traversing the dataset in order.
    image_file_name = os.path.basename(target.get('image_path'))
    output_path =  os.path.join(output_dir, image_file_name)

    ######################
    # Run the inference. #
    ######################
    try:

        sem_pred, inst_pred, sem_target, inst_target, inst_scores = infer_panoptic(img, target, mask_thresh, overlap_thresh)

        # Compute and save inst. seg. image.
        plot_panoptic_results(img, sem_pred, inst_pred, sem_target, inst_target, output_path)

        # COCO FORMAT.
        # Add image info
        coco_output["images"].append({
            "id": image_id,
            "file_name":  image_file_name,
            "height": target.get('height'),
            "width": target.get('width')
        })
        anns = instance_segmentation_to_coco(inst_pred,
                                             inst_scores,
                                             class_id=1,
                                             image_id=image_id,
                                             category_id=1,
                                             annotation_id_start=annotation_id)
        coco_output["annotations"].extend(anns)
        annotation_id += len(anns)

        print(f"Processed {output_path}")
        del img, target, sem_pred, inst_pred, sem_target, inst_target
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as E:
        print(f'WARNING: I had to skip image {output_path}, for some reason: {E}.')


    # Add categories to the COCO data.
    coco_output["categories"] = [
        {"id": 0, "name": "weird", "supercategory": "object"},
        {"id": 1, "name": "box", "supercategory": "object"},
    ]

    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(coco_output, f)
