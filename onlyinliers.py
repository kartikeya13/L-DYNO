import cv2
import numpy as np

# Load the images
img1 = cv2.imread('6.png',0)
img2 = cv2.imread('7.png',0)

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

# Draw matches with inliers in green
green = (0, 255, 0)
draw_params = dict(matchColor=green, singlePointColor=None, matchesMask=matchesMask, flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

# Save the image with inliers
cv2.imwrite('only_inliers.png', img3)

