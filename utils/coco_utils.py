import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json 

def create_semantic_mask(image_info, annotations, categories):
    """
    Create a semantic segmentation mask for a given image.

    :param image_info: Dictionary containing information about the image (id, file_name, height, width)
    :param annotations: List of annotations corresponding to the image
    :param categories: Dictionary mapping category IDs to category names
    :return: A semantic segmentation mask as a numpy array
    """
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    # Iterate through each annotation and draw the corresponding polygon on the mask
    for annotation in annotations:
        # Get the category ID and corresponding polygon
        category_id = annotation['category_id']
        polygons = annotation['segmentation']

        # Draw each polygon on the mask
        for polygon in polygons:
            # Format the polygon points for OpenCV
            poly_points = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
            # Fill the polygon on the mask with the category ID
            cv2.fillPoly(mask, [poly_points], int(category_id))

    return mask

coco_path = "./KINECT-11-13_Seg/train/_annotations.coco.json"

def get_all_images():
    pass 

    with open(coco_path) as f:

        coco_data = json.loads(f.read())

        # Convert the categories to a dictionary for easy lookup
    categories_dict = {category['id']: category['name'] for category in coco_data['categories']}

    # Process a few images (let's say the first 2) to demonstrate
    sample_images = coco_data['images']
    return sample_images

def get_mask_from_image(image):

    with open(coco_path) as f:

        coco_data = json.loads(f.read())

    categories_dict = {category['id']: category['name'] for category in coco_data['categories']}
    image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image['id']]
    semantic_mask = create_semantic_mask(image, image_annotations, categories_dict)
    
    return semantic_mask 

# sample_masks = []

# for image in sample_images:
#     # Get annotations corresponding to this image
    
#     sample_masks.append(semantic_mask)
# # Display the first couple of masks
# plt.figure(figsize=(15, 10))
# for i, mask in enumerate(sample_masks):
#     plt.subplot(1, len(sample_masks), i + 1)
#     plt.imshow(mask, cmap='nipy_spectral')
#     plt.title(f"Semantic Mask for Image ID: {sample_images[i]['id']}")
#     plt.axis('off')
# plt.show()