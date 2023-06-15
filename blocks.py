import cv2
import numpy as np
import os

def add_black_patch(images_folder, patch_width, patch_height, offset):
    # Get the list of image files in the folder
    image_files = [file for file in os.listdir(images_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        # Get the dimensions of the image
        height, width, _ = image.shape

        # Calculate the position for the top-left corner of the patch
        x_start = (width - patch_width) // 2
        y_start = height - patch_height - offset

        # Calculate the position for the bottom-right corner of the patch
        x_end = x_start + patch_width
        y_end = y_start + patch_height

        # Set the pixels within the patch region to black
        image[y_start:y_end, x_start:x_end] = [0, 0, 0]

        # Save the modified image
        output_path = os.path.join(images_folderr, image_file)
        cv2.imwrite(output_path, image)

        print(f"Processed: {image_file} -> Saved as: modified_{image_file}")

# Example usage
# Example usage
images_folder = 'kitti_full/'
images_folderr = 'k3/'
patch_width = 300  # Width of the black patch in pixels
patch_height = 80  # Height of the black patch in pixels
offset = 10  # Vertical offset from the bottom position
add_black_patch(images_folder, patch_width, patch_height, offset)



