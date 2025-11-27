import json
import os

# Training set.
file_path = '/workspace/data/Datasets/SCD_low_res_clean/annotations_trainval2017/annotations/instances_train2017.json'
image_folder = '/workspace/data/Datasets/SCD_low_res_clean/train2017/'
filenames_to_remove = [
    '',
]

# # Validation set.
# file_path = '/workspace/data/Datasets/SCD_low_res_clean/annotations_trainval2017/annotations/instances_val2017.json'
# image_folder = '/workspace/data/Datasets/SCD_low_res_clean/val2017/'
# # List of file names to remove
# filenames_to_remove = [
#     'net (9132).png',  # A lot of boxes are not labelled in GT.
#     '2293.png',        # Wrong GT: different image.
#     'w (24).png',      # Wrong GT: different image.
#     'net (6472).png',  # Wrong GT: the image and the GT have a different resolution.
#     'net (7190).png',  # Wrong GT: many boxes are not labelled.
# ]

with open(file_path) as f:
    data = json.load(f)

# Count the number of images in the JSON file.
num_samples = len(data['images'])
print(f'The JSON file contains {num_samples} images.')

# Find images matching the filenames to remove
image_ids_to_remove = set()
for image in data['images']:
    if image['file_name'] in filenames_to_remove:
        image_ids_to_remove.add(image['id'])

print(f'Found {len(image_ids_to_remove)} images to remove.')
print('Image IDs to remove:', image_ids_to_remove)

############################################################################
# REMOVE IMAGES AND ANNOTATIONS FROM JSON AND DELETE FILES.                #
############################################################################
apply_changes = False
if apply_changes:
    # Delete the image files
    print('Removing image files:')
    for image in data['images']:
        if image['id'] in image_ids_to_remove:
            image_path = os.path.join(image_folder, image['file_name'])
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f'  Deleted file {image["file_name"]}')
            else:
                print(f'  File {image["file_name"]} not found.')

    # Remove images from the dataset
    data['images'] = [image for image in data['images'] if image['id'] not in image_ids_to_remove]
    print(f'After removal, the dataset contains {len(data["images"])} images.')

    # Remove annotations for these images
    original_annotations_count = len(data['annotations'])
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in image_ids_to_remove]
    removed_annotations_count = original_annotations_count - len(data['annotations'])
    print(f'Removed {removed_annotations_count} annotations. Remaining annotations: {len(data["annotations"])}')

    # Save the updated JSON file.
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print('Updated JSON file saved.')
else:
    print('NO CHANGES APPLIED. Set apply_changes = True to modify the dataset.')