#!/usr/bin/env python3
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
import re
import time
import pandas as pd

#Dataset used:
path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/"
# path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/"
path_tau = path + "test_data_tau/tau_value"   
path_image = path + "test_data_img/training_images_318_v_1/"
# dataset_name = 'test_data_img/training_images_306_v_1'
first_image_no = 47764
last_image_no = 47906
range_frames = (first_image_no, last_image_no)
no_images = range_frames[1]- range_frames[0]

## ORB Parameters and Initialization
NUM_EXT_FEATURES = 300 
NUM_CEN_FEATURES = 150

orb_extreme = cv2.ORB_create(NUM_EXT_FEATURES)
orb_center = cv2.ORB_create(NUM_CEN_FEATURES)

## Lukas-Kanade Parameters
MIN_FEAT_THRESHOLD = 1.0
IMG_WIDTH = 1280
IMG_HEIGHT = 720
WIN_SIZE = (15,15)
MAX_LEVEL = 3
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

lk_params = dict(winSize=WIN_SIZE, maxLevel=MAX_LEVEL, criteria=CRITERIA)

## General Parameters
N_ITERATIONS = 10
IMG_WIDTH = 1280
IMG_HEIGHT = 720
XC = np.floor(IMG_WIDTH/2)
YC = np.floor(IMG_HEIGHT/2)

PERC_TTT_DISCARDED = 0.25
MIN_TTT_NUMBER = 10

def add_offset(array_kps, x_init, y_init):
    if (x_init != 0) or (y_init != 0):
        for i in range(np.size(array_kps)):
            tmp = list(array_kps[i].pt)
            tmp[0] += x_init
            tmp[1] += y_init
            array_kps[i].pt = tuple(tmp)
    return array_kps

def compute_kps(image, img_height, img_width):
    x_init_el = 0
    y_init_el = 0
    y_end_el = int(11 * img_height / 12)
    x_end_er = int(img_width)
    y_end_er = int(11 * img_height / 12)
    y_init_er = 0
    x_end_l = int(4 * img_width / 12)
    y_end_l = int(7 * img_height / 12)
    y_init_l = int(1 * img_height / 12)
    x_init_r = int(8 * img_width / 12)
    y_init_r = int(1 * img_height / 12)
    y_end_r = int(7 * img_height / 12)
    
    roi_el = image[y_init_el:y_end_el, x_init_el:x_end_l]
    roi_er = image[y_init_er:y_end_er, x_init_r:x_end_er]
    roi_c = image[y_init_l:y_end_r, x_end_l:x_init_r]
    
    keypoints = []
    keypoints = np.append(keypoints, add_offset(orb_extreme.detect(roi_el), x_init_el, y_init_el))
    keypoints = np.append(keypoints, add_offset(orb_extreme.detect(roi_er), x_init_r, y_init_er))
    keypoints = np.append(keypoints, add_offset(orb_extreme.detect(roi_c), x_end_l, y_init_l))
    
    if np.size(keypoints) > 0:
        p0 = cv2.KeyPoint_convert(keypoints)
        kps = np.float32(p0.reshape(-1, 1, 2))
        return kps
    else:
        return np.array([], dtype='f')

def tau_filtering(ttts, perc_TTT_val_discarded, min_TTT_number):
    jump = int(perc_TTT_val_discarded * np.size(ttts))
    ttts = np.sort(ttts)
    ttts = np.delete(ttts, range(jump))
    ttts = np.delete(ttts, range(np.size(ttts) - jump, np.size(ttts)))
    
    if len(ttts) >= min_TTT_number:
        ttt = np.sum(ttts) / len(ttts)
    else:
        ttt = -1

    return ttt

def read_path_from_csv(csv_file):
    # Read x, y values from CSV file and return as numpy array
    df = pd.read_csv(csv_file, header=None)
    return np.array(df)

def compute_TTT(x, y, vx, vy, img_width, img_height, xc, yc, PERC_TTT_DISCARDED, MIN_TTT_NUMBER):
    
    x_init_el_roi = 0
    y_init_el_roi = 0
    x_end_el_roi= int(3 * img_width / 12)
    y_end_el_roi= int(11 * img_height / 12)
    x_init_er_roi = int(9 * img_width / 12)
    y_init_er_roi = 0
    x_end_er_roi= int(img_width)
    y_end_er_roi= int(11 * img_height / 12)
    x_init_l_roi = int(3 * img_width / 12)
    y_init_l_roi = int(1 * img_height / 12)
    x_end_l_roi= int(5 * img_width / 12)
    y_end_l_roi= int(9.5 * img_height / 12)
    x_init_r_roi = int(7 * img_width / 12)
    y_init_r_roi = int(1 * img_height / 12)
    x_end_r_roi= int(9 * img_width / 12)
    y_end_r_roi= int(9.5 * img_height / 12)
    x_init_c_roi = int(5.5 * img_width / 12)
    y_init_c_roi = int(2.5 * img_height / 12)
    x_end_c_roi= int(6.5 * img_width / 12)
    y_end_c_roi= int(7.5 * img_height / 12)
    
    tau_right_e = []
    tau_right = []
    tau_center = []
    tau_left = []
    tau_left_e = []
    
    
    for i in range(len(x)):
        tau = (x[i]**2 + y[i]**2)**0.5 / (vx[i]**2 + vy[i]**2)**0.5
        
        # Extreme left and right
        if (x[i] >= (x_init_er_roi - xc)) and (y[i] >= (y_init_er_roi - yc)) and (y[i] <= (y_end_er_roi - yc)):
            tau_right_e = np.append(tau_right_e, tau)
        if (x[i] <= (x_end_el_roi - xc)) and (y[i] >= (y_init_el_roi - yc)) and (y[i] <= (y_end_el_roi - yc)):
            tau_left_e = np.append(tau_left_e, tau)

        # Left and right
        if (x[i] >= (x_init_r_roi - xc)) and (x[i] <= (x_end_r_roi - xc)) and (y[i] >= (y_init_r_roi - yc)) and (y[i] <= (y_end_r_roi - yc)):
            tau_right = np.append(tau_right, tau)
        if (x[i] <= (x_end_l_roi - xc)) and (x[i] >= (x_init_l_roi - xc)) and (y[i] >= (y_init_l_roi - yc)) and (y[i] <= (y_end_l_roi - yc)):
            tau_left = np.append(tau_left, tau)

        # Center
        if (x[i] >= (x_init_c_roi - xc)) and (x[i] <= (x_end_c_roi - xc)) and (y[i] >= (y_init_c_roi - yc)) and (y[i] <= (y_end_c_roi - yc)):
            tau_center = np.append(tau_center, tau)
            
    single_tau_left_e = tau_filtering(tau_left_e, PERC_TTT_DISCARDED, MIN_TTT_NUMBER)
    single_tau_left = tau_filtering(tau_left, PERC_TTT_DISCARDED, MIN_TTT_NUMBER)
    single_tau_center = tau_filtering(tau_center, PERC_TTT_DISCARDED, MIN_TTT_NUMBER)
    single_tau_right = tau_filtering(tau_right, PERC_TTT_DISCARDED, MIN_TTT_NUMBER)
    single_tau_right_e = tau_filtering(tau_right_e, PERC_TTT_DISCARDED, MIN_TTT_NUMBER)
            
    return single_tau_left_e, single_tau_left, single_tau_center, single_tau_right, single_tau_right_e



res = []
prev_image = None
curr_image = None

mean_taus_le = np.zeros(no_images)
mean_taus_l = np.zeros(no_images)
mean_taus_c = np.zeros(no_images)
mean_taus_r = np.zeros(no_images)
mean_taus_re = np.zeros(no_images)

time_array = read_path_from_csv(path + 'curr_time_hg.csv')

for it in range(N_ITERATIONS):
    
    #Print iteration status
    print(f'ITERATION {it+1}/{N_ITERATIONS}')
    
    taus_le = -1*np.ones(no_images)
    taus_l = -1*np.ones(no_images)
    taus_c = -1*np.ones(no_images)
    taus_r = -1*np.ones(no_images)
    taus_re = -1*np.ones(no_images)
    
    for idx, frame_no in enumerate(range(range_frames[0], range_frames[1])):
        
        print(frame_no)
        #Print frame status
        print(f'\tFrame {idx+1}/{last_image_no-first_image_no}', end='\r')
        
        # path_image = f'./catkin_ws/src/vision_based_navigation_ttt/{dataset_name}/{frame_no}.png'
        curr_image = cv2.imread(path_image + str(frame_no) + ".png")
        # print(curr_image)
        
        ############################ OPTICAL FLOW #############################
        kps = compute_kps(curr_image, IMG_HEIGHT, IMG_WIDTH)
        
        if (prev_image is None) or (len(prev_kps)==0):
            prev_image = curr_image
            prev_kps = kps
            continue
        assert len(prev_kps > 0)
        tracked_features, status, error = cv2.calcOpticalFlowPyrLK(prev_image, curr_image, prev_kps, None,**lk_params)
        # print('1',time_array[idx-1][0])
        # print('2',time_array[idx][0])
        DT = (time_array[idx][0]- time_array[idx-1][0]) #frame per second
        # print(FPS)
        # DT = 1/FPS #time between two consecutive frames

        # Select good points
        good_kps_new = tracked_features[status == 1]
        good_kps_old = prev_kps[status == 1]
        
        # Calculate flow field
        flow = good_kps_new - good_kps_old
        x = good_kps_old[:, 0] - XC
        y = good_kps_old[:, 1] - YC
        vx = flow[:, 0] / DT
        vy = flow[:, 1] / DT
        ########################## TIME-TO-TRANSIT ############################
        tau_le, tau_l, tau_c, tau_r, tau_re = compute_TTT(x,y,vx,vy, IMG_WIDTH, IMG_HEIGHT, XC, YC, PERC_TTT_DISCARDED, MIN_TTT_NUMBER)
        
        taus_le[idx] = tau_le
        taus_l[idx] = tau_l
        taus_c[idx] = tau_c
        taus_r[idx] = tau_r
        taus_re[idx] = tau_re
        #######################################################################
        prev_image = curr_image
        prev_kps = kps
    
    mean_taus_le += taus_le
    mean_taus_l += taus_l
    mean_taus_c += taus_c
    mean_taus_r += taus_r
    mean_taus_re += taus_re
    print('\r')

mean_taus_le /= N_ITERATIONS
mean_taus_l /= N_ITERATIONS
mean_taus_c /= N_ITERATIONS
mean_taus_r /= N_ITERATIONS
mean_taus_re /= N_ITERATIONS

np.save(path + '/of_results/OF_tau_le_hg.npy', mean_taus_le)
np.save(path + '/of_results/OF_tau_l_hg.npy', mean_taus_l)
np.save(path + '/of_results/OF_tau_c_hg.npy', mean_taus_c)
np.save(path + '/of_results/OF_tau_r_hg.npy', mean_taus_r)
np.save(path + '/of_results/OF_tau_re_hg.npy', mean_taus_re)
