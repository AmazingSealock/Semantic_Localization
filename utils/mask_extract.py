import json
from pycocotools import mask as maskUtils

# Function to decode the RLE (Run Length Encoding) mask
def decode_segmentation(segmentation):
    return maskUtils.decode(segmentation)

def extract_masks_and_classes(file_path):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']

    # Extract and decode each mask and its corresponding class
    masks = []
    classes = []
    for annotation in annotations:
        rle = {
            'size': annotation['segmentation']['size'],
            'counts': annotation['segmentation']['counts']
        }
        decoded_mask = decode_segmentation(rle)
        masks.append(decoded_mask)
        classes.append(annotation['class_name'])

    return masks, classes

import matplotlib.pyplot as plt
import numpy as np

def visualize_and_save_all_masks(masks, classes, output_file):
    # Create a blank canvas
    height, width = masks[0].shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Define colors for each class (randomly generated)
    colors = np.random.randint(0, 255, (len(classes), 3))

    # Overlay each mask on the canvas
    for mask, color in zip(masks, colors):
        red, green, blue = color
        # Update only the pixels where the mask is present
        canvas[mask == 1] = [red, green, blue]

    # Display the final image
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()

    # Save the final image
    plt.imsave(output_file, canvas)



# # Example usage
# file_path = '/root/Project/Semantic-Segment-Anything/output_kinect_11_13_data_01/KIN_0.png_semantic.json'  # Replace with your file path
# masks, classes = extract_masks_and_classes(file_path)
# print(masks,classes)
# # You can then work with 'masks' and 'classes' as needed
# # Example usage
# # Assuming you have already run the previous script to get 'masks' and 'classes'
# output_file = 'result.jpg'
# visualize_and_save_all_masks(masks, classes, output_file)
