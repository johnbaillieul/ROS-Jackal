#!/usr/bin/env python3

import cv2
import os
import numpy as np
from sklearn.linear_model import RANSACRegressor

#[674.4168821936088, 0.0, 640.5, 
# 0.0, 674.4168821936088, 360.5, 
# 0.0, 0.0, 1.0]

# Intrinsic camera matrix for the raw (distorted) images.
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]
# Projects 3D points in the camera coordinate frame to 2D pixel
# coordinates using the focal lengths (fx, fy) and principal point
# (cx, cy).

###############
# detect_corridor_direction
# This code loads an image, converts it to grayscale, applies thresholding to segment the corridor from the background, and 
# finds the contours of the thresholded image. Then it iterates through the contours, fits a line to the contour and calculates
# the slope of the line. Then it checks the slope against a threshold value to determine if the corridor is straight, 
# turning left or right. The direction of the corridor is returned as a string.
##############

def detect_corridor_direction(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Show the image
    cv2.imshow("image", image)
    cv2.waitKey(0) 
    
    # Apply thresholding to segment the corridor from the background
    ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    
    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on the image
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    # Show the image with the contours
    cv2.imshow("image", image)
    cv2.waitKey(0) 
    # Iterate through the contours
    for contour in contours:
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate the slope of the line
        slope = vy / vx
        print('sl',slope)
        
        # Set a threshold for the slope
        slope_threshold = 0.5
        
        # Check if the slope is within the threshold
        if abs(slope) < slope_threshold:
            corridor_direction = "straight"
        elif slope < -slope_threshold:
            corridor_direction = "left"
        elif slope > slope_threshold:
            corridor_direction = "right"
    
    # Return the direction of the corridor
    return corridor_direction

###############
# Here is an example of Python code that uses the OpenCV library to determine if a contour in
# an image represents a "straight", "left", or "right" direction relative to a horizontal 
# line, using the angle between the contour's major axis and the horizontal line as a 
# reference
###############

def detect_corridor_direction_wrt_horiz_ell(image_path):
    # Load image and find contours
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Show the image
    cv2.imshow("image", image)
    cv2.waitKey(0) 
    
    # Apply thresholding to segment the corridor from the background
    ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    
    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Draw the contours on the image
    cv2.drawContours(image, contour, -1, (0,255,0), 3)
    # Show the image with the contours
    cv2.imshow("image", image)
    cv2.waitKey(0) 

    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)

    # Extract the angle of the major axis
    angle = ellipse[2]

    # Define a threshold for the angle (e.g. 15 degrees)
    threshold = 5
    print(angle)
    # Check if the angle is within the threshold of the horizontal line (90 degrees)
    if abs(angle - 90) < threshold:
        corridor_direction = "straight"
    elif angle > 90:
        corridor_direction = "left"
    else:
        corridor_direction = "right"

    # Return the direction of the corridor
    return corridor_direction

#This approach uses the Hough lines method to detect lines in the image and then
#  it calculates the slope of each line. It sums the slopes of all the lines that
#  are on the left and all the lines that are on the right, and then compares the
#  values to determine if the corridor is turning left, right or is straight.

def detect_corridor_direction_wrt_horiz(image_path):
    # Load image and find contours
    image = cv2.imread(image_path)
    #Flip image horizontally
    # image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", image)
    cv2.waitKey(0) 

    imageP = np.copy(image)

    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  
    # Use the Hough lines method to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi/180, 300)

    # Iterate through the lines and draw them on the image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the image with the lines drawn on it
    cv2.imshow("Hough Lines - Standard Hough Line Transform", image)
    cv2.waitKey(0)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 300, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(imageP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # Show the image with the lines drawn on it
    cv2.imshow("Hough Lines - Probabilistic Line Transform", image)
    cv2.waitKey(0)
    # Define a threshold for the angle (e.g. 15 degrees)
    threshold = 15
    # Iterate through the lines and calculate the slope of each line
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
    
    if abs(angle) < threshold or abs(angle - 180) < threshold:
        print("Straight")
        corridor_direction = "straight"
    elif  angle < -threshold:
        print("Turning Left")
        corridor_direction = "left"
    else:
        print("Turning Right")
        corridor_direction = "right"
    return corridor_direction

def vintage_point(image_path):
    # read in the image
    img = cv2.imread(image_path)

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # use the Hough Transform to identify lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # extract the vintage point
    vintage_point = None
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # check if the line is vertical or horizontal
        if abs(x1 - x2) < 20:  # vertical line
            if vintage_point is None:
                vintage_point = (x1, (y1 + y2) // 2)
            else:
                vintage_point = ((vintage_point[0] + x1) // 2, (vintage_point[1] + (y1 + y2) // 2) // 2)
        elif abs(y1 - y2) < 20:  # horizontal line
            if vintage_point is None:
                vintage_point = ((x1 + x2) // 2, y1)
            else:
                vintage_point = ((vintage_point[0] + (x1 + x2) // 2) // 2, (vintage_point[1] + y1) // 2)

    img_center = (img.shape[1]//2, img.shape[0]//2)
    if vintage_point[0] > img_center[0] - img_center[0]/4 and vintage_point[0] < img_center[0] + img_center[0]/4:
        direction = "straight"
    if vintage_point[0] < img_center[0] - img_center[0]/4:
        direction = "left"
    elif vintage_point[0] > img_center[0]+ img_center[0]/4:
        direction = "right"

    print("The corridor is facing:", direction)


def determine_corridor_direction_vanishing(img_path, focal_length = 1000, principal_point = None):
    """
    Determines the direction of a corridor in an image using the vanishing point.

    Parameters:
    - img_path (str): The path to the image file
    - focal_length (int): The focal length of the camera used to take the image
    - principal_point (tuple): The principal point of the camera used to take the image

    Returns:
    - direction (str): The direction of the corridor ("up", "down", "left", "right")
    """

    # read in the image
    img = cv2.imread(img_path)
    
    #Flip image horizontally
    img = cv2.flip(img, 1)

    # if principal point is None, set the principal point to the center of the image
    if principal_point is None:
        principal_point = (img.shape[1]//2, img.shape[0]//2)
        
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # use the Hough Transform to identify all lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    lines_list = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines_list.append([x1, y1, x2, y2])
        
    # Create an array of midpoints and slopes of the lines
    X = np.array([((x1+x2)/2, (y1+y2)/2) for x1, y1, x2, y2 in lines_list])
    y = np.array([(x2 - x1, y2 - y1) for x1, y1, x2, y2 in lines_list])
    ransac = RANSACRegressor(min_samples=2, residual_threshold=1, max_trials=1000)
    ransac.fit(X, y)
    
    inliers = np.where(ransac.inlier_mask_ == True)[0]
    outliers = np.where(ransac.inlier_mask_ == False)[0]
    
    model_m, model_c = ransac.estimator_.coef_

    if np.isclose(model_m, 0, atol=1e-9).any() or np.isinf(model_m).any():
        vanishing_point = (int(X[inliers,0].mean()), int(y[inliers,1].mean()))
    else:
        vanishing_point = (-model_c/model_m, -model_c/model_m)
    
    vanishing_point = (vanishing_point[0],vanishing_point[1]) if len(vanishing_point) > 1 else (vanishing_point[0],0) 

    # Draw a green circle on the image at the location of the vanishing point
    cv2.circle(img, vanishing_point, 5, (0,255,0), -1)
    
    # Draw green lines on the image for the inliers lines and red lines for the outliers lines
    # for i in inliers:
    #     x1, y1, x2, y2 = lines_list[i]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # for i in outliers:
    #     x1, y1, x2, y2 = lines_list[i]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the image with lines and the vanishing point and wait for user to close the window
    cv2.imshow("lines", img)
    cv2.waitKey(0)

    # calculate the position of the vintage point relative to the center of the image
    vanishing_point_relative = (vanishing_point[0] - principal_point[0], vanishing_point[1] - principal_point[1])

    # determine the direction of the corridor based on the location of the vanishing point
    if vanishing_point_relative[1] < 0:
        direction = "up"
    elif vanishing_point_relative[1] > 0:
        direction = "down"
    if vanishing_point_relative[0] < 0:
        direction = "left"
    elif vanishing_point_relative[0] > 0:
        direction = "right"
    print(direction)
    return direction



K = np.array([[674.4168821936088, 0.0, 640.5],[0.0, 674.4168821936088, 360.5],[0.0, 0.0, 1.0]])

focal_length = np.sqrt(K[0,0]**2 + K[1,1]**2) 
principal_point = (K[0,2], K[1,2]) 

# /home/bu-robotics/catkin_ws/src/vision_based_navigation_ttt/trial_im/training_images_69_v_0.5/6729.png
# /home/bu-robotics/catkin_ws/src/vision_based_navigation_ttt/training_images/training_images_69_v_1/8121.png
images = [8155, 8126, 8166, 8364]
for i in images:
    path =  os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/training_images/training_images_69_v_1/" +str(i)+ ".png"
    # detect_corridor_direction_wrt_horiz(path)
    determine_corridor_direction_vanishing(path, focal_length = 1000, principal_point = None)

    
