import json
import os
from collections import defaultdict
from datetime import datetime


def coco_to_labelme(
    coco_json_path,
    output_dir,
    image_root=None
):
    """
    Convert a single COCO instance segmentation file to multiple LabelMe JSON files.

    Parameters
    ----------
    coco_json_path : str
        Path to COCO annotation JSON
    output_dir : str
        Directory where LabelMe JSON files will be written
    image_root : str or None
        Optional path prefix for images (LabelMe 'imagePath')
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Map category_id -> category_name
    cat_id_to_name = {
        c["id"]: c["name"] for c in coco.get("categories", [])
    }

    # Map image_id -> image info
    images = {
        img["id"]: img for img in coco.get("images", [])
    }

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    for image_id, img in images.items():
        file_name = img["file_name"]
        width = img["width"]
        height = img["height"]

        shapes = []

        for ann in anns_by_image.get(image_id, []):
            label = cat_id_to_name.get(
                ann["category_id"], "unknown"
            )

            segmentation = ann.get("segmentation", [])
            if not segmentation:
                continue

            # COCO polygon format:
            # [[x1, y1, x2, y2, ...]]
            for polygon in segmentation:
                points = [
                    [polygon[i], polygon[i + 1]]
                    for i in range(0, len(polygon), 2)
                ]

                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)

        labelme = {
            "version": "5.2.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.join(image_root, file_name)
            if image_root else file_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "createdAt": datetime.utcnow().isoformat()
        }

        out_name = os.path.splitext(file_name)[0] + ".json"
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, "w") as f:
            json.dump(labelme, f, indent=2)

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    coco_to_labelme(
        coco_json_path="/workspace/data/Datasets/fine_tuning_2025_12_19/training_annotations/annotations.json",
        output_dir="/workspace/data/Datasets/fine_tuning_2025_12_19/training_annotations_labelme/",
        image_root=None  # or e.g. "images/"
    )
