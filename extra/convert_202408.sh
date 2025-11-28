python convert_LabelMe_to_COCO_annotations.py \
    --annotations-zip "/workspace/data/Datasets/202408/annotations_labelme_validation.zip" \
    --images-zip "/workspace/data/Datasets/202408/validation.zip" \
    --annotations-path "./annotations_labelme_validation" \
    --images-path "./validation" \
    --output "/workspace/data/Datasets/202408/coco_validation.json"