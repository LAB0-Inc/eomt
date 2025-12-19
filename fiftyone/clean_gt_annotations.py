import json
import copy

# Input data.
input_path  = "/home/matteo/.lab0/data/Datasets/fine_tuning_2025_12_16/training_annotations/coco_cleaned_up_stage_1.json"
output_path = "/home/matteo/.lab0/data/Datasets/fine_tuning_2025_12_16/training_annotations/coco_cleaned_up_stage_2.json"

with open(input_path, 'r') as f:
    coco_data = json.load(f)

print("Cleaning annotations...")
print("=" * 60)

cleaned_data = copy.deepcopy(coco_data)
cleaned_annotations = []

stats = {
    'total_annotations': len(coco_data['annotations']),
    'removed_annotations': 0,
    'cleaned_annotations': 0,
    'removed_polygons_from_annotation': 0,
}

for ann in coco_data['annotations']:
    seg = ann.get('segmentation')

    # Skip annotations without segmentation
    if not seg:
        cleaned_annotations.append(ann)
        continue

    # Handle list format (polygons)
    if isinstance(seg, list):
        cleaned_polygons = []

        for polygon in seg:
            if isinstance(polygon, list):
                num_points = len(polygon) // 2

                # Keep only polygons with 3+ points
                if num_points >= 3:
                    cleaned_polygons.append(polygon)
                else:
                    stats['removed_polygons_from_annotation'] += 1
                    print(f"  Removed {num_points}-point polygon from annotation {ann.get('id', 'N/A')} (image {ann.get('image_id', 'N/A')})")

        # If we still have valid polygons, keep the annotation with cleaned polygons
        if cleaned_polygons:
            ann_copy = copy.deepcopy(ann)
            ann_copy['segmentation'] = cleaned_polygons
            cleaned_annotations.append(ann_copy)

            if len(cleaned_polygons) < len(seg):
                stats['cleaned_annotations'] += 1
        else:
            # All polygons were invalid, remove entire annotation
            stats['removed_annotations'] += 1
            print(f"    Removed entire annotation {ann.get('id', 'N/A')} (image {ann.get('image_id', 'N/A')}) - no valid polygons")

    # Handle RLE format (keep as-is)
    elif isinstance(seg, dict):
        cleaned_annotations.append(ann)

    # Unknown format, keep it
    else:
        cleaned_annotations.append(ann)

cleaned_data['annotations'] = cleaned_annotations

# Print statistics
print("\n" + "=" * 60)
print("CLEANING STATISTICS")
print("=" * 60)
print(f"Total annotations: {stats['total_annotations']}")
print(f"Cleaned annotations (had some polygons removed): {stats['cleaned_annotations']}")
print(f"Completely removed annotations: {stats['removed_annotations']}")
print(f"Total polygons removed: {stats['removed_polygons_from_annotation']}")
print(f"Final annotations: {len(cleaned_annotations)}")

# Save cleaned annotations
with open(output_path, 'w') as f:
    json.dump(cleaned_data, f)

print(f"\n✓ Cleaned annotations saved to: {output_path}")
print("\nNow try loading with FiftyOne using the cleaned file!")

