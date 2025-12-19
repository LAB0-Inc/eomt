import fiftyone as fo
dataset = fo.load_dataset("FT1_3_on_val")

results = dataset.evaluate_detections(
    "pred_with_bbs",
    gt_field="gt_with_bbs",
    eval_key="inst_eval",
    use_masks=True,
    compute_mAP=True,
)

# Print the report.
results.print_report()
print(results.mAP())
