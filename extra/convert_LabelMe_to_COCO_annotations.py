#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations (one per image) to a single COCO format JSON file.

Usage:
    python convert_LabelMe_to_COCO_annotations.py --annotations-zip annotations.zip --images-zip images.zip --output coco_annotations.json

    Or if annotations are in a folder within the zip:
    python convert_LabelMe_to_COCO_annotations.py --annotations-zip annotations.zip --annotations-path ./annotations_labelme_validation --output coco_annotations.json
"""

import json
import zipfile
import argparse
from pathlib import Path
from datetime import datetime


def convert_labelme_to_coco(
    annotations_zip_path,
    images_zip_path=None,
    annotations_path_in_zip="./",
    images_path_in_zip="./",
    output_json_path="coco_annotations.json",
    category_mapping=None,
):
    """
    Convert LabelMe annotations to COCO format.

    Args:
        annotations_zip_path: Path to zip file containing JSON annotations
        images_zip_path: Path to zip file containing images (optional, for getting image dimensions)
        annotations_path_in_zip: Path within zip where annotations are located
        images_path_in_zip: Path within zip where images are located
        output_json_path: Output path for COCO JSON file
        category_mapping: Dict mapping LabelMe label strings/ints to COCO category IDs
    """

    # Default category mapping if none provided
    if category_mapping is None:
        category_mapping = {
            "0": 0,
            "1": 1,
            "box": 1,
            "Box": 1,
            0: 0,
            1: 1,
        }

    # Initialize COCO structure
    coco_output = {
        "info": {
            "description": "Converted from LabelMe format",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories from mapping
    unique_categories = set(category_mapping.values())
    for cat_id in sorted(unique_categories):
        coco_output["categories"].append({
            "id": cat_id,
            "name": f"class_{cat_id}",
            "supercategory": "object"
        })

    annotation_id = 1
    image_id = 1

    # Open annotations zip
    with zipfile.ZipFile(annotations_zip_path, 'r') as ann_zip:
        # Get all JSON files in the specified path
        json_files = [
            f for f in ann_zip.namelist()
            if f.startswith(str(annotations_path_in_zip).lstrip('./'))
            and f.endswith('.json')
            and not Path(f).name.startswith('.')
        ]

        print(f"Found {len(json_files)} JSON annotation files")

        # Open images zip if provided
        img_zip = None
        if images_zip_path:
            img_zip = zipfile.ZipFile(images_zip_path, 'r')

        for json_file in sorted(json_files):
            try:
                # Load LabelMe annotation
                with ann_zip.open(json_file) as f:
                    labelme_data = json.load(f)

                # Get image filename
                if "imagePath" in labelme_data:
                    img_filename = Path(labelme_data["imagePath"]).name
                else:
                    # Assume JSON filename matches image filename
                    img_filename = Path(json_file).stem + ".jpg"

                # Get image dimensions
                width = labelme_data.get("imageWidth")
                height = labelme_data.get("imageHeight")

                # If dimensions not in JSON, try to get from image
                if (width is None or height is None) and img_zip:
                    try:
                        from PIL import Image
                        img_path = str(Path(images_path_in_zip) / img_filename)
                        with img_zip.open(img_path) as img_file:
                            img = Image.open(img_file)
                            width, height = img.size
                    except:
                        print(f"Warning: Could not get dimensions for {img_filename}, skipping")
                        continue

                if width is None or height is None:
                    print(f"Warning: No dimensions available for {img_filename}, skipping")
                    continue

                # Add image entry
                coco_output["images"].append({
                    "id": image_id,
                    "file_name": img_filename,
                    "width": width,
                    "height": height,
                })

                # Process shapes/annotations
                if "shapes" in labelme_data:
                    for shape in labelme_data["shapes"]:
                        # Get label and map to category
                        label = shape.get("label", "0")
                        if label != 'box':
                            print(f'Found interesting label: {label}')

                        # Try both string and int versions of label
                        if label in category_mapping:
                            category_id = category_mapping[label]
                        elif isinstance(label, str) and label.isdigit():
                            int_label = int(label)
                            if int_label in category_mapping:
                                category_id = category_mapping[int_label]
                            else:
                                print(f"Warning: Label '{label}' not in mapping, skipping")
                                continue
                        else:
                            print(f"Warning: Label '{label}' not in mapping, skipping")
                            continue

                        # Get polygon points
                        points = shape.get("points", [])
                        if not points:
                            continue

                        # Convert to COCO segmentation format (flat list)
                        segmentation = []
                        for point in points:
                            segmentation.extend([float(point[0]), float(point[1])])

                        # Calculate bounding box
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min

                        # Add annotation
                        coco_output["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [segmentation],
                            "area": bbox_width * bbox_height,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "iscrowd": 0,
                        })

                        annotation_id += 1

                image_id += 1

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

        if img_zip:
            img_zip.close()

    # Write output
    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Images: {len(coco_output['images'])}")
    print(f"Annotations: {len(coco_output['annotations'])}")
    print(f"Categories: {len(coco_output['categories'])}")
    print(f"Output saved to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to COCO format"
    )
    parser.add_argument(
        "--annotations-zip",
        required=True,
        help="Path to zip file containing LabelMe JSON annotations"
    )
    parser.add_argument(
        "--images-zip",
        help="Path to zip file containing images (optional, for getting dimensions)"
    )
    parser.add_argument(
        "--annotations-path",
        default="./",
        help="Path within annotations zip where JSON files are located"
    )
    parser.add_argument(
        "--images-path",
        default="./",
        help="Path within images zip where image files are located"
    )
    parser.add_argument(
        "--output",
        default="coco_annotations.json",
        help="Output path for COCO format JSON file"
    )
    parser.add_argument(
        "--category-mapping",
        help='JSON string for category mapping, e.g., \'{"0": 0, "1": 1}\''
    )

    args = parser.parse_args()

    # Parse category mapping if provided
    category_mapping = None
    if args.category_mapping:
        category_mapping = json.loads(args.category_mapping)

    convert_labelme_to_coco(
        annotations_zip_path=args.annotations_zip,
        images_zip_path=args.images_zip,
        annotations_path_in_zip=args.annotations_path,
        images_path_in_zip=args.images_path,
        output_json_path=args.output,
        category_mapping=category_mapping,
    )