import fiftyone as fo
from fiftyone import ViewField as F

# dataset = fo.load_dataset("FT1_3_on_val")
dataset = fo.load_dataset("fine_tuning")

# # If you want to see only a filtered version of the samples.
# LOW_IOU = 0.90
# HIGH_FP_AREA = 0.2
# HIGH_FN_AREA = 0.2
# condition_1 = F("iou") < LOW_IOU
# condition_2 = F("FP_area") > HIGH_FP_AREA
# condition_3 = F("FN_area") > HIGH_FN_AREA
# combined_view = dataset.match(condition_1 | condition_2 | condition_3)
# # combined_view = dataset.match(condition_2 | condition_3)  # Only FP or FN.
# print(f"Number of samples selected =  {len(combined_view)}")
# session = fo.launch_app(combined_view)

# If you want to see all the samples.
sorted_view = dataset.sort_by("TP_area", reverse=True)

session = fo.launch_app(sorted_view)

# Keep the session alive - this blocks the script
session.wait()
