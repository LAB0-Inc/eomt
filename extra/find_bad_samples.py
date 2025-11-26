import csv
import matplotlib.pyplot as plt
import numpy as np

input_csv_path = '/workspace/data/EOMT/Output/val_run2_on_train_AVOID_OVERWRITE/iou_log.csv'
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

indices = [1206, 1258, 1577, 1814, 2219, 2597, 2599, 2721, 2790, 2808, 2833, 2939, 3236, 3242, 3312, 3683, 3954, 4735, 5201, 5638, 5720]

for index in indices:
    print(data[index][1][10:])
# # Count samples for IoU ranges
# iou_thresholds = [0.0, 0.5, 0.7, 0.9, 1.0]
# counts = {f"{iou_thresholds[i]}-{iou_thresholds[i+1]}": 0 for i in range(len(iou_thresholds)-1)}
# for _, _, iou, _, _ in data:
#     for i in range(len(iou_thresholds)-1):
#         if iou_thresholds[i] <= iou < iou_thresholds[i+1]:
#             counts[f"{iou_thresholds[i]}-{iou_thresholds[i+1]}"] += 1
#             break

# print("IoU Range Counts:")
# for range_key, count in counts.items():
#     print(f"  {range_key}: {count}")

# # bad_samples = [d for d in data if d[2] < 0.9]
# # print(f"Number of bad samples (IoU < 0.5): {len(bad_samples)}")

# # Sort by IoU
# data.sort(key=lambda x: x[2])

# pass

# # IoU Range Counts:
# #   0.0-0.5: 348
# #   0.5-0.7: 380
# #   0.7-0.9: 524
# #   0.9-1.0: 12826