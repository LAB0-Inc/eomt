import json

# Load predictions
with open("/home/matteo/.lab0/data/EOMT/Output/val_on_ft_training/predictions.json", 'r') as f:
    pred_data = json.load(f)

# Load ground truth to get all images
with open("/home/matteo/.lab0/data/Datasets/fine_tuning_2025_12_16/training_annotations/annotations_cleaned.json", 'r') as f:
    gt_data = json.load(f)

print("Analyzing predictions coverage...")
print("=" * 60)

# Get all image IDs from ground truth
all_image_ids = set(img['id'] for img in gt_data['images'])
image_id_to_filename = {img['id']: img['file_name'] for img in gt_data['images']}

# Get image IDs that have predictions
predicted_image_ids = set(ann['image_id'] for ann in pred_data['annotations'])

# Find images without predictions
images_without_predictions = all_image_ids - predicted_image_ids

# Count predictions per image
predictions_per_image = {}
for ann in pred_data['annotations']:
    img_id = ann['image_id']
    predictions_per_image[img_id] = predictions_per_image.get(img_id, 0) + 1

# Print results
print(f"\nTotal images in ground truth: {len(all_image_ids)}")
print(f"Images with predictions: {len(predicted_image_ids)}")
print(f"Images WITHOUT predictions: {len(images_without_predictions)}")

if images_without_predictions:
    print("\n" + "=" * 60)
    print("IMAGES WITHOUT PREDICTIONS")
    print("=" * 60)
    for img_id in sorted(images_without_predictions):
        filename = image_id_to_filename.get(img_id, 'Unknown')
        print(f"  Image ID {img_id}: {filename}")

# Show distribution of predictions per image
print("\n" + "=" * 60)
print("PREDICTIONS PER IMAGE DISTRIBUTION")
print("=" * 60)

if predictions_per_image:
    from collections import Counter

    pred_counts = Counter(predictions_per_image.values())

    print(f"\nImages with 0 predictions: {len(images_without_predictions)}")
    for count in sorted(pred_counts.keys()):
        print(f"Images with {count} prediction(s): {pred_counts[count]}")

    # Show some examples
    print("\n" + "=" * 60)
    print("EXAMPLES OF IMAGES WITH PREDICTIONS")
    print("=" * 60)
    for img_id, count in list(predictions_per_image.items())[:10]:
        filename = image_id_to_filename.get(img_id, 'Unknown')
        print(f"  Image ID {img_id} ({filename}): {count} prediction(s)")
else:
    print("No predictions found in the file!")

# Check if predictions file has the expected structure
print("\n" + "=" * 60)
print("PREDICTIONS FILE STRUCTURE")
print("=" * 60)
print(f"Total annotations in predictions: {len(pred_data['annotations'])}")
print(f"Keys in predictions file: {list(pred_data.keys())}")

if len(pred_data['annotations']) > 0:
    print("\nFirst prediction sample:")
    first_pred = pred_data['annotations'][0]
    print(f"  Keys: {list(first_pred.keys())}")
    print(f"  Image ID: {first_pred.get('image_id', 'N/A')}")
    print(f"  Category ID: {first_pred.get('category_id', 'N/A')}")
    print(f"  Has segmentation: {'segmentation' in first_pred}")
    print(f"  Has bbox: {'bbox' in first_pred}")
    print(f"  Has score: {'score' in first_pred}")