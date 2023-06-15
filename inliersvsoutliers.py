'''
import cv2
import numpy as np

# Load the images
img1 = cv2.imread('masked_image_0.png',0)
img2 = cv2.imread('masked_image_1.png',0)

# Initialize the ORB detector and find keypoints and descriptors
orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features using the Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)

# Store the keypoints from the good matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

# Use RANSAC to estimate a homography matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Convert mask to a list of booleans and then to integers
matchesMask = mask.ravel().tolist()
matchesMask = [int(val) for val in matchesMask]

# Count inliers and outliers
inliers = matchesMask.count(1)
outliers = matchesMask.count(0)

# Print number of inliers and outliers
print('Number of inliers: ', inliers)
print('Number of outliers: ', outliers)

# Draw matches with inliers in green and outliers in red
green = (0, 255, 0)
red = (0, 0, 255)

# Draw inliers in green
draw_params = dict(matchColor=green, singlePointColor=None, matchesMask=matchesMask, flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

# Draw outliers in red
outliersMask = [not int(val) for val in matchesMask]
outliersMask = [int(val) for val in outliersMask]
draw_params['matchColor'] = red
draw_params['matchesMask'] = outliersMask
img4 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

# Combine both images for final output
final_output = cv2.addWeighted(img3, 0.7, img4, 0.3, 0)

# Display result
cv2.imwrite('InliersandOutliers.png', final_output)
'''


'''
import cv2
import numpy as np
import os

# Folder path containing the images
folder_path = 'knew/'

# Get the list of image files in the folder
image_files = os.listdir(folder_path)
image_files.sort()  # Sort the files in ascending order

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=2000)

# Process image pairs
for i in range(400):
    # Load the images
    img1 = cv2.imread(os.path.join(folder_path, image_files[i]), 0)
    img2 = cv2.imread(os.path.join(folder_path, image_files[i + 1]), 0)

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match features using the Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Store the keypoints from the good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate a homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Separate inliers and outliers
    inlier_points = []
    outlier_points = []
    for m, mask_val in zip(matches, mask.ravel().tolist()):
        if mask_val == 1:
            inlier_points.append(kp2[m.trainIdx].pt)
        else:
            outlier_points.append(kp2[m.trainIdx].pt)

    # Print number of inliers and outliers
    num_inliers = len(inlier_points)
    num_outliers = len(outlier_points)
    print(num_inliers,num_outliers)
    
    #print('Number of outliers:', num_outliers)

    # Draw inliers as green points
    img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for point in inlier_points:
        cv2.circle(img_out, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

    # Draw outliers as red points
    for point in outlier_points:
        cv2.circle(img_out, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    # Save the result
    output_file = str(i) + '.png'
    cv2.imwrite(output_file, img_out)


'''

import cv2
import numpy as np
import os

# Folder path containing the images
folder_path = 'knew/'

# Get the list of image files in the folder
image_files = os.listdir(folder_path)
image_files.sort()  # Sort the files in ascending order

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=2000)

# Process image pairs
for i in range(len(image_files) - 1):
    # Load the images
    img1 = cv2.imread(os.path.join(folder_path, image_files[i]), 0)
    img2 = cv2.imread(os.path.join(folder_path, image_files[i + 1]), 0)

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match features using the Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Store the keypoints from the good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate a homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Separate inliers and outliers
    inlier_points = []
    outlier_points = []
    for m, mask_val in zip(matches, mask.ravel().tolist()):
        if mask_val == 1:
            inlier_points.append(kp2[m.trainIdx].pt)
        else:
            outlier_points.append(kp2[m.trainIdx].pt)

    # Print number of inliers and outliers
    num_inliers = len(inlier_points)
    num_outliers = len(outlier_points)
    print('Number of inliers:', num_inliers)
    print('Number of outliers:', num_outliers)

    # Draw inliers as green points
    img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for point in inlier_points:
        cv2.circle(img_out, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

    # Draw outliers as red points
    for point in outlier_points:
        cv2.circle(img_out, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    # Save the result
    output_file = str(i) + '.png'
    cv2.imwrite(output_file, img_out)




