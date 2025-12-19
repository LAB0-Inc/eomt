# In case MongoDB gets stuck, you can clear it like this, BUT YOU WILL LOSE SAVED DATASETS. There must be better ways.
# rm -rf ~/.fiftyone/var/lib/mongo


import fiftyone as fo
import fiftyone.utils.labels as foul
import numpy as np
import os


def fix_bbox_to_mask_dimensions(sample_collection, field_name):
    """
    TODO: CHECK THIS: IT WAS GENERATED AUTOMATICALLY.
    Adjust bounding boxes to match mask dimensions.

    Args:
        sample_collection: FiftyOne dataset or view
        field_name: Name of the detections field (e.g., "pred_with_bbs" or "gt_with_bbs")
    """
    modified_count = 0
    total_detections = 0

    for sample in sample_collection.iter_samples(progress=True):
        field = sample[field_name]
        if field is None:
            continue

        img_h, img_w = sample.metadata.height, sample.metadata.width
        modified = False

        for det in field.detections:
            total_detections += 1

            if det.mask is None:
                continue

            # Get current bounding box (in relative coordinates [x, y, w, h])
            bbox = det.bounding_box

            # Get mask dimensions
            mask_h, mask_w = det.mask.shape

            # Calculate what the bbox dimensions should be based on mask
            bbox_w_from_mask = mask_w / img_w
            bbox_h_from_mask = mask_h / img_h

            # Check if adjustment is needed
            if not np.isclose(bbox[2], bbox_w_from_mask, rtol=1e-5) or \
               not np.isclose(bbox[3], bbox_h_from_mask, rtol=1e-5):

                # Update bounding box dimensions to match mask
                # Keep the top-left corner (x, y) the same
                new_bbox = [bbox[0], bbox[1], bbox_w_from_mask, bbox_h_from_mask]
                det.bounding_box = new_bbox
                modified = True
                modified_count += 1

        if modified:
            sample[field_name] = field
            sample.save()

    print(f"\nSummary:")
    print(f"Total detections processed: {total_detections}")
    print(f"Bounding boxes adjusted: {modified_count}")
    return modified_count


def change_gt_object_class(detections):
    for det in detections.detections:
        det.label = "box"
    return

def compute_mask_area_fraction(detection, image_height, image_width):
    """Compute the area of a detection's mask as a fraction of total image area."""
    if detection.mask is None:
        return 0.0
    mask_array = detection.mask
    # Count pixels in the mask
    mask_pixel_count = np.sum(mask_array)
    # Total image pixels
    total_pixels = image_height * image_width

    return mask_pixel_count / total_pixels

def compute_binary_iou(gt_mask, pred_mask):
    """
    Compute IoU for binary segmentation (one class + background).

    Args:
        gt_mask: Ground truth mask (H, W) - typically 0 for background, 255 for object
        pred_mask: Prediction mask (H, W) - typically 0 for background, 255 for object

    Returns:
        iou: Intersection over Union score
    """
    # Binarize masks (in case values are 255 instead of 1)
    gt_binary = gt_mask > 0
    pred_binary = pred_mask > 0

    # Compute intersection and union
    intersection = np.sum(gt_binary & pred_binary)
    union = np.sum(gt_binary | pred_binary)

    # Compute IoU
    iou = intersection / union if union > 0 else 0.0

    return iou


# 1. Create the dataset, load images and GT labels.
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="/home/matteo/.lab0/data/Datasets/202408/validation_images/",
    # labels_path="/home/matteo/.lab0/data/Datasets/202408/validation_annotations/annotations.json",  # ORIGINAL
    labels_path="/home/matteo/.lab0/data/Datasets/202408/validation_annotations/coco_validation_v_2025_12_15.json",
    label_types="segmentations",
    label_field="gt_with_bbs",
    name="FT1_3_on_val"  # Name of the dataset.
)


# 2. Load predictions.
prediction_path = '/home/matteo/.lab0/data/EOMT/Output/val_FT_3_on_202408_val/'
dataset.merge_samples(
    fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path="/home/matteo/.lab0/data/Datasets/202408/validation_images/",
        labels_path=prediction_path + "/predictions_mask_th_0.8_overlap_th_0.8.json",
        label_types="segmentations",
        label_field="pred_with_bbs"
    )
)


# 3. Ensure that BBs fit their masks in size, otherwise mAP computation fails.
fix_bbox_to_mask_dimensions(dataset, "pred_with_bbs")
fix_bbox_to_mask_dimensions(dataset, "gt_with_bbs")

# 4. Change class name for GT.
for sample in dataset:
    change_gt_object_class(sample.gt_with_bbs)  # Set object type to "box".
    sample.save()


# 5. Create labels without BBs for the visualization.
#    This is done by transforming the type of detection to a segmentation. This discards the score of the detections and more.
current_field = "gt_with_bbs"
new_segmentation_field = "gt"
foul.objects_to_segmentations(
    dataset,
    current_field,
    new_segmentation_field)
current_field = "pred_with_bbs"
new_segmentation_field = "pred"
foul.objects_to_segmentations(
    dataset,
    current_field,
    new_segmentation_field)

# 6. Evaluate detections and GT segments: classify things as TP/FP/FN.
results = dataset.evaluate_detections(
    "pred_with_bbs",       # predictions field
    gt_field="gt_with_bbs",
    eval_key="eval",       # the field where evaluation results will be stored
    iou=0.5,               # IoU threshold for matching masks
    classwise=True,        # compute metrics per class
    backend="masks"        # Use the segments, not the BBs.
)

# 7. Compute the F1 score for each image.
sample_results = results.samples
for sample in dataset:
    r = sample_results[sample.id]
    tp = r["eval_tp"]
    fp = r["eval_fp"]
    fn = r["eval_fn"]
    if tp == 0 and fp == 0 and fn == 0:
        f1 = 0.0
    else:
        f1 = 2*tp / (2*tp + fp + fn)
    sample["F1"] = f1
    sample.save()

# 8. Compute the Semantic Segmentation IoU for each image.
for sample in dataset:
    iou = compute_binary_iou(sample['gt']['mask'], sample['pred']['mask'])
    sample['iou'] = iou
    sample.save()



# 9. Create labels for False Positives and Missed Detections (FN)
for sample in dataset:
    fp_dets = [det for det in sample["pred_with_bbs"].detections if det.eval == "fp"]
    sample["FP_with_bbs"] = fo.Detections(detections=fp_dets)

    # Compute total FP area as fraction of image
    fp_area = sum(compute_mask_area_fraction(det, sample.metadata.height, sample.metadata.width) for det in fp_dets)
    sample["FP_area"] = float(fp_area * 100)  # We save it as a percentage for better visualization on the app.

    # False Negatives.
    fn_dets = [det for det in sample["gt_with_bbs"].detections if det.eval == "fn"]
    sample["FN_with_bbs"] = fo.Detections(detections=fn_dets)

    # Compute total FN area as fraction of image
    fn_area = sum(compute_mask_area_fraction(det, sample.metadata.height, sample.metadata.width) for det in fn_dets)
    sample["FN_area"] = float(fn_area * 100)  # We save it as a percentage for better visualization on the app.
    sample.save()

# 10. Create a version of FPs and FNs without BBs, for the visualization.
#    This is done by transforming the type of detection to a segmentation. This discards the score of the detections and more.
current_field = "FP_with_bbs"
new_segmentation_field = "FP"
foul.objects_to_segmentations(
    dataset,
    current_field,
    new_segmentation_field)
current_field = "FN_with_bbs"
new_segmentation_field = "FN"
foul.objects_to_segmentations(
    dataset,
    current_field,
    new_segmentation_field)



# 11. Set up the app.
# This allows you to sort by other variables as well.
sorted_view = dataset.sort_by("F1", reverse=True)

# Set custom colors. THIS DOES NOT WORK.
# dataset.app_config.color_scheme = fo.ColorScheme(
#     color_pool=[
#         "#FF0000",  # Red
#         "#00FF00",  # Green
#         "#0000FF",  # Blue
#         "#FFFF00",  # Yellow
#         "#FF00FF",  # Magenta
#         "#00FFFF",  # Cyan
#     ],
#     fields=[
#         {"path": "FP_with_bbs", "color": "#FF0000"},
#         {"path": "FN_with_bbs", "color": "#00FF00"},
#         {"path": "FP_area", "color": "#FF0000"},
#         {"path": "FN_area", "color": "#00FF00"},
#         {"path": "eval_fp", "color": "#FF0000"},
#         {"path": "eval_fn", "color": "#00FF00"},
#         {"path": "gt_with_bbs", "color": "#0000FF"},
#         {"path": "pred_with_bbs", "color": "#FFFF00"},
#         {"path": "gt", "color": "#0000FF"},
#         {"path": "pred", "color": "#FFFF00"},
#     ]
# )

# Save the dataset to MongoDB.
dataset.save()

# Export data to CSV file.
# Add filename field to all samples
for sample in dataset:
    sample["filename"] = os.path.basename(sample.filepath)
    sample.save()
dataset.export(
    export_dir=prediction_path+"/per_sample_analysis.csv",
    dataset_type=fo.types.CSVDataset,
    fields=["filename", "iou", "FP_area", "FN_area"])

# 12. Launch the app.
# session = fo.launch_app(dataset)
session = fo.launch_app(sorted_view)

# Keep the session alive - this blocks the script
session.wait()
