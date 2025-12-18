python convert_LabelMe_to_COCO_annotations.py \
    --annotations-zip "/workspace/data/Datasets/fine_tuning_2025_12_16/training_annotations_labelme_cleaned_up_stage_1.zip" \
    --images-zip "/workspace/data/Datasets/fine_tuning_2025_12_16/training_images.zip" \
    --annotations-path "./" \
    --images-path "./" \
    --output "/workspace/data/Datasets/fine_tuning_2025_12_16/coco_cleaned_up_stage_1.json"
