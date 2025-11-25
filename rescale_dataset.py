import json
import os
from PIL import Image
from pathlib import Path


def scale_coco_dataset(
    json_path,
    images_dir,
    output_json_dir,
    output_json_path,
    output_images_dir,
    target_width=640,
    target_height=640
):
    """
    Scale COCO format dataset images and adjust annotations accordingly.

    Args:
        json_path: Path to input COCO JSON file
        images_dir: Directory containing input images
        output_json_path: Path for output JSON file
        output_images_dir: Directory for scaled images
        target_width: Target width for scaling calculation (default: 640)
        target_height: Target height for scaling calculation (default: 480)
    """

    # Load COCO JSON
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories.
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    # Create a mapping of image_id to image info
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Process each image
    print(f"Processing {len(coco_data['images'])} images...")
    for img_index, img_info in enumerate(coco_data['images']):
        if img_index % 100 == 0:
            print(f"  Processing image {img_index + 1}/{len(coco_data['images'])}.")
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)

        # Open image
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print(f"Warning: Image {img_filename} not found, skipping...")
            continue

        try:
            original_width = img_info['width']
            original_height = img_info['height']

            # Calculate scaling factor so that the largest dimension matches the desired value.
            # (The image will later have to be padded, to make the smallest dimension also match the desired value).
            width_factor = target_width / original_width
            height_factor = target_height / original_height
            factor = min(width_factor, height_factor)  # Min factor = largest scaling.

            # Calculate new dimensions
            new_width = int(original_width * factor)
            new_height = int(original_height * factor)

            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        except Exception as e:
            print(f"Error processing image {img_filename}: {e}, skipping...")
            continue

        try:
            # Save the output in a lossless format.
            filename_without_ext = os.path.splitext(img_filename)[0]
            new_filename = f"{filename_without_ext}.png"

            # Save scaled image in lossless format
            output_path = os.path.join(output_images_dir, new_filename)
            resized_img.save(output_path)

            # Update image info in JSON
            img_info['file_name'] = new_filename
            img_info['width'] = new_width
            img_info['height'] = new_height

            # Store factor for annotation scaling
            image_id_to_info[img_id]['scale_factor'] = factor
        except Exception as e:
            print(f"Error saving image {img_filename}: {e}, skipping...")
            continue

    # Process annotations
    print(f"Processing {len(coco_data['annotations'])} annotations...")
    for ann in coco_data['annotations']:
        img_id = ann['image_id']

        if img_id not in image_id_to_info:
            continue

        factor = image_id_to_info[img_id].get('scale_factor', 1.0)

        # Scale bounding box [x, y, width, height]
        if 'bbox' in ann:
            bbox = ann['bbox']
            ann['bbox'] = [
                bbox[0] * factor,  # x
                bbox[1] * factor,  # y
                bbox[2] * factor,  # width
                bbox[3] * factor   # height
            ]

        # Scale segmentation polygons
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                # Polygon format: list of [x1,y1,x2,y2,...]
                scaled_segmentation = []
                for polygon in ann['segmentation']:
                    scaled_polygon = [coord * factor for coord in polygon]
                    scaled_segmentation.append(scaled_polygon)
                ann['segmentation'] = scaled_segmentation
            elif isinstance(ann['segmentation'], dict):
                # RLE format - needs special handling
                # For RLE, we'd need to decode, scale, and re-encode
                # This is complex, so we'll just scale the bbox for now
                print(f"Warning: RLE segmentation format detected for annotation {ann['id']}, only bbox scaled")

        # Scale area
        if 'area' in ann:
            ann['area'] = ann['area'] * (factor ** 2)

    # Remove temporary scale_factor from image info
    for img_info in coco_data['images']:
        img_info.pop('scale_factor', None)

    # Save updated JSON
    print(f"Saving updated JSON to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    # Validation.
    output_json_dir="/workspace/data/Datasets/SCD_low_res/annotations_trainval2017/annotations/"
    scale_coco_dataset(
        json_path="/workspace/data/Datasets/SCD_for_EoMT/annotations_trainval2017/annotations/instances_val2017.json",
        images_dir="/workspace/data/Datasets/SCD_for_EoMT/val2017",
        output_json_dir=output_json_dir,
        output_json_path=f"{output_json_dir}/instances_val2017.json",
        output_images_dir="/workspace/data/Datasets/SCD_low_res/val2017"
    )

    # Training.
    output_json_dir="/workspace/data/Datasets/SCD_low_res/annotations_trainval2017/annotations/"
    scale_coco_dataset(
        json_path="/workspace/data/Datasets/SCD_for_EoMT/annotations_trainval2017/annotations/instances_train2017.json",
        images_dir="/workspace/data/Datasets/SCD_for_EoMT/train2017",
        output_json_dir=output_json_dir,
        output_json_path=f"{output_json_dir}/instances_train2017.json",
        output_images_dir="/workspace/data/Datasets/SCD_low_res/train2017"
    )

