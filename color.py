import cv2
import numpy as np
import os

def paint_road_sky(images_folder):
    # Get the list of image files in the folder
    image_files = [file for file in os.listdir(images_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        # Define the color values for road and sky
        road_color = [0, 255, 0]  # Green
        sky_color = [0, 0, 255]  # Red

        # Create a mask for road and sky regions
        mask = np.zeros_like(image)
        mask[:image.shape[0] // 2, :] = road_color
        mask[image.shape[0] // 2:, :] = sky_color

        # Apply the mask to the image
        painted_image = cv2.addWeighted(image, 0.8, mask, 0.2, 0)

        # Save the modified image
        output_path = os.path.join(images_folder, 'modified_' + image_file)
        cv2.imwrite(output_path, painted_image)

        print(f"Processed: {image_file} -> Saved as: modified_{image_file}")

# Example usage
images_folder = 'kitti_full/'
paint_road_sky(images_folder)

