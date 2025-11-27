import csv
import matplotlib.pyplot as plt
import numpy as np

# input_csv_path = '/workspace/data/EOMT/Output/val_run2_on_train_GOOD/iou_log.csv'
input_csv_path = '/workspace/data/EOMT/Output/val_run2_on_val_GOOD/iou_log.csv'

data = []
with open(input_csv_path, 'r') as f:
    reader = csv.reader(f)
    # next(reader)  # Skip header
    for row in reader:
        batch_idx = int(row[0])
        image_path = row[1]
        iou = float(row[2])
        n_detections = int(row[3])
        n_gt_masks = int(row[4])
        data.append((batch_idx, image_path, iou, n_detections, n_gt_masks))


# Count samples for IoU ranges
iou_thresholds = [0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
counts = {f"{iou_thresholds[i]}-{iou_thresholds[i+1]}": 0 for i in range(len(iou_thresholds)-1)}
for _, _, iou, _, _ in data:
    for i in range(len(iou_thresholds)-1):
        if iou_thresholds[i] <= iou < iou_thresholds[i+1]:
            counts[f"{iou_thresholds[i]}-{iou_thresholds[i+1]}"] += 1
            break

print("IoU Range Counts:")
for range_key, count in counts.items():
    print(f"  {range_key}: {count}")

# Sort by IoU
data.sort(key=lambda x: x[2])

# # Print 40 samples with worst IoU.
# for index, entry in enumerate(data[0:60]):
#     print(f"{index}: {entry}")

pass
