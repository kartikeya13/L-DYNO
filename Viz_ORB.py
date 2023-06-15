import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pykitti

## LOAD IMU BASED 3x4 TRANSFORMATIONS#
#dataset_path = 'k/'
dataset_path = 'k/'
dataset_pose_path = "00.txt"
## Camera intrinsic paramters ##
k = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
     [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]], dtype=np.float32)
## Min feature number to track ## 
kMinNumFeature = 3000
## Create empty image to draw trajectory ## 
traj = np.zeros((1500, 1500, 3), dtype=np.uint8)
x_loc = []
z_loc = []
cur_R = None
cur_t = None
basedir = 'imu_dataset'
date = '2011_09_30'
drive = '0034'
#dataset = pykitti.raw(basedir, date, drive,frames=range(0, 11555, 5))
def Read_dataset(dataset_path):
    seq00_path = dataset_path
    seq00_list = [seq00_path+f for f in listdir(seq00_path) if isfile(join(seq00_path, f))]
    seq00_list.sort()
    return seq00_list

def Read_gt_trajectory(dataset_pose_path):
    file_09 = open(dataset_pose_path,"r") 
    lines = file_09.readlines()
    x = []
    y = []
    z = []
    for i in lines:
        x.append(i.split(' ')[0])
        y.append(i.split(' ')[1])
        z.append(i.split(' ')[2])
    file_09.close()
    gt_trajectory =  np.stack((x, y, z)).astype(np.float32)
    return gt_trajectory

def getAbsoluteScale(gt_trajectory, frame_id):  
    x_prev = float(gt_trajectory[0, frame_id-1])
    y_prev = float(gt_trajectory[1, frame_id-1])
    z_prev = float(gt_trajectory[2, frame_id-1])
    x = float(gt_trajectory[0, frame_id])
    y = float(gt_trajectory[1, frame_id])
    z = float(gt_trajectory[2, frame_id])
    return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))


def featureTracking(image_ref, image_cur, px_ref):
    lk_params = dict(winSize  = (21, 21), 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params) 
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]
    return kp1, kp2


def process_first_frames(first_frame, second_frame, k):
    det = cv2.FastFeatureDetector_create()
    kp1 = det.detect(first_frame)
    kp1 = np.array([x.pt for x in kp1], dtype=np.float32)

    kp1, kp2 = featureTracking(first_frame, second_frame, kp1)
    E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
    kp1 = kp2
    return kp1, R, t
def calculate_mean_trajectory_error(est_trajectory, gt_trajectory):
    # Ensure both trajectories are numpy arrays
    est_trajectory = np.array(est_trajectory)
    gt_trajectory = np.array(gt_trajectory)
    
    # Ensure both trajectories have same shape
    assert est_trajectory.shape == gt_trajectory.shape, "Estimated and Ground Truth trajectories should have same shape"

    # Calculate Euclidean distances for each point
    error_distances = np.sqrt(np.sum((est_trajectory - gt_trajectory) ** 2, axis=1))

    # Return mean error
    return np.mean(error_distances)

seq00_list = Read_dataset(dataset_path)
gt_trajectory = Read_gt_trajectory(dataset_pose_path)
first_frame = cv2.imread(seq00_list[0], 0)
second_frame = cv2.imread(seq00_list[1], 0)
kp1, cur_R, cur_t = process_first_frames(first_frame, second_frame, k)
last_frame = second_frame
def visualizeMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image where the matches will be drawn
    out_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

    # Draw the images side by side
    out_img[:img1.shape[0], :img1.shape[1]] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    out_img[:img2.shape[0], img1.shape[1]:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw the matches
    for match in matches:
        # Get the keypoints for the current match
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Get the coordinates of the keypoints
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a line connecting the keypoints
        cv2.line(out_img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (0, 255, 0), 1)

    return out_img

    return out_img
## main loop ## 
for i in range(200):
    print("i:",i)
    ## read the new frame from the image paths list ## 
    #if i == 1000:
    #    break


    new_frame = cv2.imread(seq00_list[i], 0)
    ## track the feature movement from prev frame to current frame ## 
    kp1, kp2 = featureTracking(last_frame, new_frame, kp1)
    det = cv2.orb_create() #.detect
    des1 = det.detect(last_frame, None)
    des2 = det.detect(new_frame, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    match_image = visualizeMatches(last_frame, kp11, new_frame, kp22, good_matches)

    # Save the matches image
    matches_filename = "matches_frame{}.jpg".format(i)
    #cv2.imwrite(matches_filename, match_image)
    #cv2.imwrite("matches.jpg", match_image)
    ## find the rotation and translation matrix ##
    E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
    ## find the change of the feature location ## 
    change = np.mean(np.abs(kp2 - kp1))
    ## find the scale of the movemnt from the ground truth trajectory ## 
    absolute_scale = getAbsoluteScale(gt_trajectory, i)
    if absolute_scale > 2 :
        absolute_scale = 1
    ## check if the vehicle not moving by check the change value ## 
    if change > 5:
       ## accumulate the translation and rotation to find the X, Y, Z locations ## 
        cur_t = cur_t + absolute_scale * cur_R.dot(t)
        cur_R = R.dot(cur_R)
    ## if the number of detect features below threshold value recaulc the feature ## 
    if(kp1.shape[0] < kMinNumFeature):
        det = cv2.FastFeatureDetector_create()
        kp2 = det.detect(new_frame)
        kp2 = np.array([x.pt for x in kp2], dtype=np.float32)
    ## Get ready for the next loop ##
    kp1 = kp2
    last_frame = new_frame
    # start after the first two frames ##
    if i > 2 :
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0.0, 0.0, 0.0
    ## save x, z loc ##
    x_loc.append(x)
    z_loc.append(z)
    good_matches = []
    draw_x, draw_y = int(x)+300, int(z)+220
    true_x, true_y = int(gt_trajectory[0, i])+300, int(gt_trajectory[2, i])+220
    cv2.circle(traj, (draw_x,draw_y), 1, (0,0,255), 1)
    #cv2.circle(traj, (imuX,imuY), 1, (255,255,255), 3)
    #cv2.circle(traj, (imuX,imuY), 1, (255,255,255), 3)
    cv2.circle(traj, (true_x,true_y), 1, (0,255,0), 2)
    cv2.rectangle(1000, (600, 400), (900, 90), (0,0,0), -1)

    cv2.putText(traj, text1, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv2.putText(traj, text2, (20,80), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv2.imshow('Road facing camera', new_frame)
    cv2.imshow('Trajectory', traj)
    # Close the frame
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Combine x and z locations to form estimated trajectory
est_trajectory = np.vstack((x_loc, z_loc)).T

# Get the ground truth trajectory
gt_trajectory = gt_trajectory[[0, 2]].T

# Assume `estimated_trajectory` and `ground_truth_trajectory` are lists of 3D positions
# (x, y, z), and they have the same length.

def calculate_average_trajectory_error(estimated_trajectory, ground_truth_trajectory):
    estimated_values = estimated_trajectory
    actual_values = ground_truth_trajectory
    assert len(estimated_trajectory) == len(ground_truth_trajectory)
    
    estimated_trajectory = np.array(estimated_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory)
    
    error_values = estimated_trajectory - ground_truth_trajectory
    squared_errors = np.square(error_values)
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    return rmse
#ate = calculate_average_trajectory_error(np.array(x_loc), np.array(z_loc))
ate = calculate_average_trajectory_error((finalx,finalz),(finaltx,finaltz))
ate = round(ate, 1) 
print("Average Trajectory Error:", ate)	
#ate = calculate_average_trajectory_error(x_loc, z_loc)
#print("Average Trajectory Error:", ate)

# Calculate mean trajectory error
'''
# Release and Destroy
cv2.destroyAllWindows()
cv2.imwrite('ORB_output.png', traj)
## Plot Result ##
plt.figure(figsize=(8, 8), dpi=100)
plt.title("X Z Trajectory")
plt.ylabel("X")
plt.xlabel("Z")
plt.plot(x_loc, z_loc, label="Trajectory")
plt.plot(gt_trajectory[0], gt_trajectory[2], label="GT-Trajectory")
plt.legend()
plt.show()
'''
