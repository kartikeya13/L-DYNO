import cv2
import numpy as np
import os

def convert_blue_to_black(images_folder):
    # Get the list of image files in the folder
    image_files = [file for file in os.listdir(images_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for blue color in HSV
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Create a mask for blue color range
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Convert the blue regions to black
        image[mask > 0] = [0, 0, 0]

        # Save the modified image
        output_path = os.path.join(images_folderr, 'modified_' + image_file)
        cv2.imwrite(output_path, image)

        print(f"Processed: {image_file} -> Saved as: modified_{image_file}")

# Example usage
images_folderr = 'k3/'
images_folder = 'kitti_full/'
convert_blue_to_black(images_folder)

