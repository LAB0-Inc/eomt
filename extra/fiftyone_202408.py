# In case MongoDB gets stuck:
# rm -rf ~/.fiftyone/var/lib/mongo


import fiftyone as fo
import fiftyone.utils.labels as foul


def delete_attribute(detections, attribute):
    for det in detections.segmentations:
        del(det[attribute])
    return

def change_gt_object_class(detections):
    for det in detections.detections:
        det.label = "box"
    return

# 1. Create the dataset, load images and GT labels.
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="/home/matteo/.lab0/data/Datasets/202408/validation_images/",
    labels_path="/home/matteo/.lab0/data/Datasets/202408/validation_annotations/annotations.json",
    label_types="segmentations",
    label_field="gt_with_bbs",
    name="FT1_3_on_val"  # Name of the dataset.
)

# 2. Load predictions.
dataset.merge_samples(
    fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path="/home/matteo/.lab0/data/Datasets/202408/validation_images/",
        labels_path="/home/matteo/.lab0/data/EOMT/Output/val_FT_3_on_202408_val/predictions.json",
        label_types="segmentations",
        label_field="pred_with_bbs"
    )
)

# 3. Change class name for GT.
for sample in dataset:
    change_gt_object_class(sample.gt_with_bbs)  # Set object type to "box".
    sample.save()

# 4. Remove BBs from the visualization.
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

# 5. Evaluate detections and GT segments: classify things as TP/FP/FN.
results = dataset.evaluate_detections(
    "pred_with_bbs",       # predictions field
    gt_field="gt_with_bbs",
    eval_key="eval",       # the field where evaluation results will be stored
    iou=0.5,               # IoU threshold for matching masks
    classwise=True,        # compute metrics per class
    backend="masks"        # Use the segments, not the BBs.
)

# 6. Compute the F1 score for each image.
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


# 7. Create labels for False Positives and Missed Detections (FN)
for sample in dataset:
    fp_dets = [det for det in sample["pred_with_bbs"].detections if det.eval == "fp"]
    sample["FP_with_bbs"] = fo.Detections(detections=fp_dets)

    # False Negatives.
    fn_dets = [det for det in sample["gt_with_bbs"].detections if det.eval == "fn"]
    sample["FN_with_bbs"] = fo.Detections(detections=fn_dets)

    sample.save()

# 8. Create a version of FPs and FNs without BBs, for the visualization.
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

# 9. Set up the app.
# This allows you to sort by other variables as well.
sorted_view = dataset.sort_by("F1", reverse=True)

# 10. Launch the app.
# session = fo.launch_app(dataset)
session = fo.launch_app(sorted_view)

# Keep the session alive - this blocks the script
session.wait()
