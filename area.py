import os
import cv2
import numpy as np

def calculate_black_masks_area(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the total area variables
    total_area = 0
    image_area = image.shape[0] * image.shape[1]

    # Iterate over each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        total_area += area

        # Draw the contour on the image (highlight in green)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    return total_area, image_area, image

# Folder path containing the images
folder_path = 'knew/'

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct the full image path
        image_path = os.path.join(folder_path, filename)

        # Calculate the area of black masks, the total area of the image, and get the annotated image
        black_masks_area, image_area, annotated_image = calculate_black_masks_area(image_path)

        # Print the results
        #print("File: {}".format(filename))
        print(int(black_masks_area),",")
        #print("Total area of the image: {} pixels".format(image_area))

        # Display the annotated image
        #cv2.imshow("Annotated Image", annotated_image)
        #cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

