import cv2
import numpy as np
import os

# Folder path containing the images
folder_path = 'knew/'

# Get the list of image files in the folder
image_files = os.listdir(folder_path)
image_files.sort()  # Sort the files in ascending order

# Custom color map for bright red and black
color_map = np.zeros((256, 1, 3), dtype=np.uint8)
color_map[:, 0, 2] = np.arange(256)  # Red channel
color_map[:, 0, 1] = 0  # Green channel
color_map[:, 0, 0] = 0  # Blue channel
folder_pathh = 'heat/'
# Iterate over each image file
for image_file in image_files:
    # Load the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary mask of visible portions
    _, thresholded = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Convert the thresholded image to single channel grayscale
    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

    # Create a heat map by applying the custom color map
    heat_map = cv2.LUT(thresholded, color_map)

    # Overlay the heat map on the original image
    overlaid = cv2.addWeighted(image, 0.7, heat_map, 0.3, 0)

    # Save the image with the visible portion as a heat map
    output_path = os.path.join(folder_pathh, 'heatmap_' + image_file)
    cv2.imwrite(output_path, overlaid)

cv2.destroyAllWindows()

