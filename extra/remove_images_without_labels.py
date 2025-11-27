import json
import os

# Training set.
# file_path = '/workspace/data/Datasets/SCD_low_res_clean/annotations_trainval2017/annotations/instances_train2017.json'
# image_folder = '/workspace/data/Datasets/SCD_low_res_clean/train2017/'

# Validation set.
file_path = '/workspace/data/Datasets/SCD_low_res_clean/annotations_trainval2017/annotations/instances_val2017.json'
image_folder = '/workspace/data/Datasets/SCD_low_res_clean/val2017/'

with open(file_path) as f:
    data = json.load(f)

# Count the number of images in the JSON file.
num_samples = len(data['images'])
print(f'The JSON file contains {num_samples} images.')

# Count the number of images refereced by the annotations.
image_ids = set()
for annotation in data['annotations']:
    image_ids.add(annotation['image_id'])

print(f'The annotations refer to {len(image_ids)} unique images.')

# Find the images without annotations.
all_image_ids = set(image['id'] for image in data['images'])
images_without_annotations = all_image_ids - image_ids
print(f'Number of images without annotations: {len(images_without_annotations)}')
print('Images without annotations IDs:', images_without_annotations)

# Chech whether the images with annotation exist on the disk.
missing_images = []
for image in data['images']:
    if image['id'] in image_ids:
        image_path = os.path.join(image_folder, image['file_name'])
        if not os.path.exists(image_path):
            missing_images.append(image['file_name'])
print(f'Number of images with annotations that do not have an image file: {len(missing_images)}')
print('Missing images:', missing_images)

############################################################################
# REMOVE IMAGES WITHOUT ANNOTATIONS FROM THE DATASET AND DELETE THE FILES. #
############################################################################
apply_changes = False
if apply_changes:
    # Delete the image files without annotations.
    print('Images without annotations file names:')
    for image in data['images']:
        if image['id'] in images_without_annotations:
            if os.path.exists(os.path.join(image_folder, image['file_name'])):
                os.remove(os.path.join(image_folder, image['file_name']))
                print(f'  Deleted file {image["file_name"]}')
            else:
                print(f'  File {image["file_name"]} not found.')

    # Remove images without annotations from the dataset
    data['images'] = [image for image in data['images'] if image['id'] not in images_without_annotations]
    print(f'After removal, the dataset contains {len(data["images"])} images.')

    # Save the updated JSON file.
    with open(file_path, 'w') as f:
        json.dump(data, f)
else:
    print('CHANGES WERE NOT APPLIED BECAUSE apply_changes IS False')


