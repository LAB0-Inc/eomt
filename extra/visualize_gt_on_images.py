import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' if you have Qt installed
import matplotlib.pyplot as plt
import cv2
import random
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from os import makedirs

def generate_bgr_colors(n):
    """
    Generate n fully saturated, equally spaced colors along the hue circle in BGR.

    Args:
        n (int): Number of colors.

    Returns:
        List of n colors in BGR format (values 0-255).
    """
    # Equally spaced hues [0, 180) for OpenCV's HSV (H ranges 0-179)
    hues = np.linspace(0, 179, n, endpoint=False).astype(int)

    # Full saturation and value
    s = np.full_like(hues, 255)
    v = np.full_like(hues, 255)

    # Stack into HSV array
    hsv_colors = np.stack([hues, s, v], axis=1).astype(np.uint8)

    # Convert to BGR using OpenCV
    bgr_colors = cv2.cvtColor(hsv_colors[np.newaxis, :, :], cv2.COLOR_HSV2BGR)[0]

    # Return as a list of tuples
    return [tuple(int(c) for c in color) for color in bgr_colors]



# Images with wrong dimensions (train):
# [{'index': 11096, 'file_name': 'net (957).jpg'}, {'index': 11516, 'file_name': 'net (2954).jpg'}, {'index': 12517, 'file_name': 'net (4114).jpg'}, {'index': 14077, 'file_name': 'net (11006).jpg'}]
# Images with no annotations (train):
# [{'index': 1386, 'file_name': 'IMG_20200917_142551.jpg'}, {'index': 1935, 'file_name': '2938.jpg'}, {'index': 4090, 'file_name': '82.jpg'}, {'index': 6896, 'file_name': 'net (13004).jpg'}, {'index': 7028, 'file_name': 'net (6405).jpg'}, {'index': 7344, 'file_name': 'net (665).jpg'}, {'index': 7531, 'file_name': 'net (8577).jpg'}, {'index': 8023, 'file_name': 'net (4403).jpg'}, {'index': 8184, 'file_name': 'net (2006).jpg'}, {'index': 8297, 'file_name': 'net (12506).jpg'}, {'index': 8637, 'file_name': 'net (4007).jpg'}, {'index': 9013, 'file_name': 'net (9182).jpg'}, {'index': 10173, 'file_name': 'net (904).jpg'}, {'index': 10211, 'file_name': 'net (4016).jpg'}, {'index': 10999, 'file_name': 'net (2003).jpg'}, {'index': 11430, 'file_name': 'net (11398).jpg'}, {'index': 11593, 'file_name': 'net (2941).jpg'}, {'index': 11851, 'file_name': 'net (12548).jpg'}, {'index': 11943, 'file_name': 'net (3325).jpg'}, {'index': 12091, 'file_name': 'net (1909).jpg'}, {'index': 12301, 'file_name': 'net (3511).jpg'}, {'index': 13025, 'file_name': 'net (11800).jpg'}, {'index': 13171, 'file_name': 'net (9181).jpg'}, {'index': 13322, 'file_name': 'net (2218).jpg'}, {'index': 13433, 'file_name': 'net (3555).jpg'}, {'index': 13837, 'file_name': 'net (4012).jpg'}]

# Images with no annotations (validation):
# [{'index': 1042, 'file_name': 'net (4019).jpg'}, {'index': 1273, 'file_name': 'net (9125).jpg'}, {'index': 1356, 'file_name': 'net (8060).jpg'}, {'index': 1603, 'file_name': 'net (2001).jpg'}, {'index': 1984, 'file_name': 'net (16065).jpg'}]

train_or_val = 'train'
root_folder = '/home/matteo/.lab0/data/Datasets/LSCD_for_EoMT/'
ann_file = root_folder + f'/annotations_trainval2017/annotations/instances_{train_or_val}2017.json'
img_dir = root_folder + f'/{train_or_val}2017'
out_dir = root_folder + f'/visualization_{train_or_val}2017'


images_with_wrong_dimensions = []
images_with_no_annotations = []
makedirs(out_dir, exist_ok=True)
coco = COCO(ann_file)
img_ids = coco.getImgIds()
first_image_index = 0
for image_index in range(first_image_index, len(img_ids)):
    wrong_dim = False
    img_id = img_ids[image_index]
    img_info = coco.loadImgs(img_id)[0]
    # print(str(image_index) + ' / ' + str(len(img_ids)) + ' ' + img_info['file_name'])
    img_path = f"{img_dir}/{img_info['file_name']}"
    image = cv2.imread(img_path)  # keep as BGR, uint8.

    if (not (image.shape[0] == img_info['height']) or
        not (image.shape[1] == img_info['width'])):
        print('Wrong dimensions!')
        images_with_wrong_dimensions.append({'index':image_index, 'file_name':img_info['file_name']})
        print(str(image_index) + ' / ' + str(len(img_ids)) + ' ' + img_info['file_name'])
        print(f'Height = {img_info['height']}')
        print(f'Width = {img_info['width']}')
        wrong_dim = True
        continue

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    if len(anns)  < 1:
        print('Empty annotations!')
        images_with_no_annotations.append({'index':image_index, 'file_name':img_info['file_name']})
        continue


    #######################################
    # Overlay the segments on the images. #
    #######################################

    # Scale the images, so that it doesn't take too long to perform the overlays.
    max_size = 900  # set max display width/height in pixels
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    if scale < 1:  # only shrink large images
        new_w, new_h = int(w * scale), int(h * scale)
        image_display = cv2.resize(image, (new_w, new_h))
        # for ann in anns:
        #     ann['segmentation'] = (np.array(ann['segmentation']) * scale).tolist()
        #     # TODO: I DO NOT SCALE BB's BECAUSE I DON'T VISUALIZE THEM.
    else:
        scale = 1
        image_display = image

    colors = list(np.random.permutation(generate_bgr_colors(len(anns))))
    for index, ann in enumerate(anns):
        color = colors[index]
        mask = coco.annToMask(ann)
        if scale < 1:  # only shrink large images
            mask = cv2.resize(mask, (new_w, new_h))

        # Alpha blending
        alpha = 0.5
        for c in range(3):
            image_display[:, :, c] = np.where(mask==1,
                                    (image_display[:, :, c]*(1-alpha) + alpha*color[c]).astype(np.uint8),
                                    image_display[:, :, c])

        # # Draw bounding box
        # x, y, w, h = ann['bbox']
        # cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

    out_file_path = f"{out_dir}/{img_info['file_name']}"
    cv2.imwrite(out_file_path, image_display)

print('Images with wrong dimensions:')
print(images_with_wrong_dimensions)

print('Images with no annotations:')
print(images_with_no_annotations)
