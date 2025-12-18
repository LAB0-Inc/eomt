python convert_LabelMe_to_COCO_annotations.py \
    --annotations-zip "/workspace/data/Datasets/202408/OriginalAnnotationFormat/annotations_labelme_validation_v_2025_12_15.zip" \
    --images-zip "/workspace/data/Datasets/202408/validation_images.zip" \
    --annotations-path "./annotations_labelme_validation_v_2025_12_15" \
    --images-path "./validation" \
    --output "/workspace/data/Datasets/202408/coco_validation_v_2025_12_15.json"