import json
import os

# SCD DATASET.
# # Training set.
# file_path = '/workspace/data/Datasets/SCD_low_res_clean/annotations_trainval2017/annotations/instances_train2017.json'
# image_folder = '/workspace/data/Datasets/SCD_low_res_clean/train2017/'
# filenames_to_remove = [
#     'net (7945).png',  # Weird image with books, rather than boxes.
#     'net (391).png',   # Wrong GT.
#     'y (147).png',     # Wrong GT.
#     'net (7714).png',  # Confusing image: it has paper bags that look like boxes, but are not labelled.
#     '2021.png',        # Few boxes, very far.
#     'net (7195).png',  # Confusing image: it has box-like things that are not labelled.
#     'net (7202).png',  # Many small boxes are not labelled
#     'net (5237).png',  # Confusing: not clear if big object is a box or not.
#     'net (15899).png', # Big oil cans look like boxes, but are not labelled as such.
#     'net (7221).png',  # Many box-like objects are not labelled.
#     'net (9098).png',  # Many boxes are not labelled.
#     '257.png',         # One tote is not labelled.
#     '4 (106).png',     # Big boxes at the back are not annotated.
#     'net (4084).png',  # Many boxes are not labelled.
#     'net (14639).png', # Not clear if a large thing is a box or not.
#     'net (480).png',   # Many boxes/faces are not annotated.
#     'net (7209).png',  # Many boxes are not labelled.
#     'net (7223).png',  # Many boxes are not labelled.
#     'net (9988).png',  # Many boxes are not labelled.
#     'net (9038).png',  # Many boxes are not labelled.
#     'IMG_20200917_094504.png', # Confusing: one large thing is not a box.
#     'net (3840).png',  # One large box is not labelled.
#     '1 (28).png',      # One large box is not labelled.
#     'l (111).png',     # Many boxes are not labelled.
#     'net (3030).png',  # Many boxes are not labelled.
#     '2089.png',        # One big box is not labelled.
#     'net (11439).png', # One big box is not labelled.
#     'net (5184).png',  # Many boxes are not labelled.
#     '3 (359).png',     # Bad GT.
#     'l (112).png',     # Many boxes are not labelled.
#     'net (597).png',   # Many boxes are not labelled.
#     'net (9850).png',  # Many boxes are not labelled.
#     '1 (56).png',      # Many boxes are not labelled.
#     'net (5750).png',  # Many boxes are not labelled.
#     'net (6575).png',  # Confusing: some boxes behind plexiglass, some labels are on reflections.
#     'net (7711).png',  # Many boxes are not labelled.
#     'net (15578).png', # Many boxes are not labelled.
#     'net (582).png',   # Many boxes are not labelled.
#     'net (7814).png',  # Many boxes are not labelled.
#     'net (7033).png',  # Many boxes are not labelled.
#     'net (9811).png'   # Many boxes are not labelled.
# ]

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

# Fine-tuning DATASET.
# Training set.
file_path = '/workspace/data/Datasets/fine_tuning_2025_12_16/training_annotations/coco_cleaned_up_stage_1.json'
image_folder = '/workspace/data/Datasets/fine_tuning_2025_12_16/training_images/'
# List of file names to remove
filenames_to_remove = [
    'cam_left_down_1751994841453970637.jpg',  # Falling box.
    'cam_left_down_1752181451540315760.jpg',  # Falling box.
    'cam_left_down_1752182050967694861.jpg',  # Falling box.
    'cam_left_down_1751919590369908239.jpg',  # Falling box.
    'cam_left_down_1750974360144962328.jpg',  # Falling box.
    'cam_left_down_1751566344207034581.jpg',  # Boxes are very far, they are too small on the image to annotate them.
]

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