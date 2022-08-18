#!/usr/bin/python
#
# =======================================================
# Computing L-K Optical Flow from Sparse Visual Features
# =======================================================
#
# =======================================================
# Built with:
# =======================================================
# Python                 3.7.6
# opencv-python          3.4.2.17
# *opencv-contrib-python 3.4.2.17 (*required only if using patented algorithms SIFT or SURF)
# numpy                  1.16.4
# --------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------
# This script implements Lucas-Kanade (L-K) optical flow computing from a video stream using the ORB feature detector.
#
# ORB, the feature detector, can be used interchangeably with other feature detectors provided in OpenCV (SIFT, SURF,
# etc.) In that case, some parameters, such as the ratio test or the number of output matches, need to be modified
# accordingly. For details, please refer to the document describing the OpenCV implementation of vision algorithms.
#
# The implementation described in this script is a memoryless version, in the sense that the optical flow computation
# of every frame is independent. It can be extended to include a memory storing detected features and their descriptors
# that are updated periodically (updating every frame or every a few frames or based on other asynchronous criteria).
# By continuously tracking features across multiple frames, the accuracy of estimated optical flow velocities and
# robustness of perception can be enhanced.
#
# Various filters are optional for enhancing the accuracy of estimated optical flow velocities, such as the built-in
# ratio test provided in the OpenCV, median filtering in local neighborhoods, outlier rejection or smoothness filter.
# --------------------------------------------------------------------------------------------------------------------

# =======================================================
# Updated: Jul. 19, 2021
# =======================================================
#
# --------------------------------------------------------------------------------------------------------------------


# =======================================================
# Import
# =======================================================
import cv2
import numpy as np
import traceback
import sys
#
# --------------------------------------------------------------------------------------------------------------------


# =======================================================
# Input Settings
# =======================================================
# The input video stream can be from either a video file or camera live stream.
#
# ****************************
# Choose ONE of the following
# ****************************
# If the video stream is from a local file, e.g., sample.mov, in the same folder with the script,
name = 'sample.mov'
cap = cv2.VideoCapture(name)
#
# Only extract and process one frame in every a few frames to accommodate processing speed or other purposes (e.g.,
# avoiding high frequency random noises); frames in between will be discarded.
interval = 3
# interval = n means processing one frame in every n frames
# interval = 1 means processing every frame in the video.
#
# --------------OR--------------
# If the video stream is from a live camera, e.g., camera 0,
# cap = cv2.VideoCapture(0)
# interval = 1
# ***************************
#
# --------------------------------------------------------------------------------------------------------------------


# =======================================================
# Output Settings
# =======================================================
# Visualization of computed optical flow field and statistics marked on the original video stream is recorded as a
# .mov format video file.

fps = 25    # set the frame rate for the output video
#
# --------------------------------------------------------------------------------------------------------------------


# =======================================================
# Initialization
# =======================================================
# Read a frame to initialize the algorithm, set corresponding parameters, and open output writers.

# read the first frame
ret, img = cap.read()

# track the index of the frame
frame_no = 1
print(frame_no)

k = 10    # control gain, used for calculating the steering signal
# resize the image frame to the desired scale
# usually for reducing the image size to accommodate processing speed
img = cv2.resize(img, (0,0), fx = 1.0, fy = 1.0)

# read the width and height parameters of the image frame
(w, h) = (img.shape[1], img.shape[0])

img1 = img    # initialize the first image to be processed

# initialize output video writer
vout = cv2.VideoWriter()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vout.open('output.mov',fourcc, fps, (w, h), True)
# the video is only saved when the writer is released

# open a log writer to output raw data (if needed)
f = open('log.txt', 'w')

# record the parameters for the video in the log file
f.write('\n')
f.write(name+', '+str(fps)+', '+str(interval)+', '+"\n")
# the log is only saved when the writer is released

#
# --------------------------------------------------------------------------------------------------------------------


# =======================================================
# Main Algorithm
# =======================================================
# The main function will loop forever until being manually halted (e.g., keyboard interrupt) or reaching the end of
# the input video file. When using live camera stream as input, the algorithm has to be terminated manually; when using
# a video file as input, the algorithm can either automatically terminate when it reaches the end of the video or be
# terminated manually.
#
# To ensure video and file writers are always appropriately released (to save the output files), the main function runs
# inside the try-block. When the algorithm is terminated (either manually or automatically), the except-block always
# has a chance to execute, which will save the output files and print out the reason the program exits.

try:
    while 1:
        # read and resize the second frame for processing; the first frame is always the frame read last time
        for i in range(interval):
            ret, img2 = cap.read()
            frame_no = frame_no + 1
            img2 = cv2.resize(img2, (0, 0), fx=1.0, fy=1.0)

        # initialize image matrices
        img_show = np.zeros_like(img)  # img_show contains the image for final output
        mask_fig = np.zeros_like(img)  # mask_fig contains visualization of optical-flow vectors and statistics

        # Examine if the end of video file is reached or an exception if encountered.
        # Terminate the algorithm and print the reason if the images are not legitimate.
        if img1 is None or img2 is None:
            print('Could not open or find the images!')
            # exit(0)
            break

        # -- Step 1: Detect the keypoints using a feature detector, compute the descriptors
        # SIFT and SURF can be used interchangeably.
        # Parameters for ORB are slightly different.

        # ****************************
        # Choose ONE of the following
        # ****************************
        # *** SIFT ***
        #
        # detector = cv2.xfeatures2d.SIFT_create()
        # keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        # keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
        #
        # -- Step 2: Matching descriptor vectors with a FLANN based matcher
        # Since SIFT is a floating-point descriptor, NORM_L2 is used
        # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # knn_matcher: Finds the k best matches for each descriptor from a query set.
        # knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
        #
        # -- Filter matches using the Lowe's ratio test
        # ratio_thresh = 0.7
        # good_matches = []
        # for m, n in knn_matches:
        #     if (m.distance < ratio_thresh * n.distance):
        #         good_matches.append(m)
        #
        # --------------OR--------------
        # *** SURF ***
        #
        # minHessian = 1000    # minHessian: Threshold for hessian keypoint detector used in SURF.
        # detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
        # keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        # keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
        #
        # -- Step 2: Matching descriptor vectors with a FLANN based matcher
        # Since SURF is a floating-point descriptor, NORM_L2 is used
        # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # knn_matcher: Finds the k best matches for each descriptor from a query set.
        # knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
        #
        # -- Filter matches using the Lowe's ratio test
        # ratio_thresh = 0.7
        # good_matches = []
        # for m, n in knn_matches:
        #     if (m.distance < ratio_thresh * n.distance):
        #         good_matches.append(m)
        #
        # --------------OR--------------
        # *** ORB ***
        #
        # nfeatures: number of best features to retain
        detector = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
        #
        # -- Step 2: Matching descriptor vectors with a brute-force matcher
        # Since ORB is a binary-string descriptor, NORM_HAMMING is used
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if (len(descriptors1) != 0) & (len(descriptors2) != 0):
            matches = bf.match(descriptors1, descriptors2)
        else:
            continue
        # Select best matches and discard the rest
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:3000]
        #
        # ***************************

        # ‘good_matches' stores a list of matches, which are pairs of indexes of the keypoints in the lists.
        # For convenience, create another variable 'plain_matches', which directly stores the "keypoint" objects.
        plain_match = []

        for i in good_matches:
            plain_match.append([i.queryIdx, i.trainIdx])

        # 'pnt_pair' stores the locations of the keypoints in the frame in each match.
        pnt_pair = []

        for m, n in plain_match:
            pnt_pair.append([keypoints1[m].pt, keypoints2[n].pt])

        # Evenly segment the image frame into 10 vertical panels. Create variables to store the statistics of
        # optical-flow velocities in each panel for later calculation

        sum_seg = [0] * 10    # the sum of optical-flow speed in a panel
        cnt_seg = [0] * 10    # the number of optical-flow vectors in a panel
        avg_seg = [0] * 10    # average optical-flow speed calculated as sum_seg[i]/cnt_seg[i]
        sum_ori_seg = [0] * 10    # the sum of orientation of optical-flow velocities in a panel
        avg_ori_seg = [0] * 10    # average orientation of optical-flow velocities
        sum_hor_seg = [0] * 10    # the sum of horizontal OF speed in a panel
        avg_hor_seg = [0] * 10    # average horizontal OF speed in a panel

        # calculate the statistics of OF velocities for all pairs of matched features in 'pnt_pair' for filtering
        ii = 0
        for i in pnt_pair:
            p0 = (int(i[0][0]), int(i[0][1]))    # get the location of the first feature in the match
            p1 = (int(i[1][0]), int(i[1][1]))    # get the location of the second feature in the match
            dx = (p1[0] - p0[0])    # calculate the horizontal speed of the optical-flow
            dy = (p1[1] - p0[1])    # calculate the vertical speed of the optical-flow
            seg_class = int(((p1[0]-p0[0])/2+p0[0]) / w * 10)    # find the panel this OF vector located in
            cnt_seg[seg_class] = cnt_seg[seg_class] + 1    # count the number of OF vectors in the panel
            sum_seg[seg_class] = sum_seg[seg_class] + np.sqrt(dx * dx + dy * dy)    # sum up OF speed
            sum_ori_seg[seg_class] = sum_ori_seg[seg_class] + np.arctan(dy/(dx+0.000001))    # sum up OF orientation
            ii = ii + 1

        # calculate the average OF speed and orientation in each panel
        # set the average to zero if no OF vector in the panel
        for i in range(10):
            if cnt_seg[i] == 0:
                avg_seg[i] = 0
            else:
                avg_seg[i] = sum_seg[i]/cnt_seg[i]
                avg_ori_seg[i] = sum_ori_seg[i]/cnt_seg[i]

        flow_vec = []    # create a container for OF FILTERED vectors
        avg_tau_seg = [0] * 10    # average time-to-transit for each panel

        ii = 0
        for i in pnt_pair:
            p0 = (int(i[0][0]), int(i[0][1]))    # get the location of the first feature in the match
            p1 = (int(i[1][0]), int(i[1][1]))    # get the location of the second feature in the match
            dx = (p1[0] - p0[0])    # calculate the horizontal speed of the optical-flow
            dy = (p1[1] - p0[1])    # calculate the vertical speed of the optical-flow
            seg_class = int(((p1[0]-p0[0])/2+p0[0]) / w * 10)    # find the panel this OF vector located in
            # -- OF filtering
            # Reject the OF vector if its speed is greater than 1.2x average OF speed in the panel.
            if np.sqrt(dx*dx+dy*dy) > avg_seg[seg_class]*1.2:
                cnt_seg[seg_class] = cnt_seg[seg_class] - 1
                continue
            # Reject the OF vector if its speed is greater than some threshold of the panel
            if np.sqrt(dx*dx+dy*dy) > abs(seg_class-4.5)*(w/50) + (w/50):
                cnt_seg[seg_class] = cnt_seg[seg_class] - 1
                continue
            # Reject the OF vector if its orientation differs more than pi/4 from averaged OF orientation in the panel
            if abs(np.arctan(dy/(dx+0.00001)) - avg_ori_seg[seg_class]) > np.pi/4:
                cnt_seg[seg_class] = cnt_seg[seg_class] - 1
                continue
            # sum up FILTERED horizontal OF speed
            sum_hor_seg[seg_class] = sum_hor_seg[seg_class] + dx
            color = (0, 255, 0)
            # draw on FILTERED OF vectors on the original image for visualization
            cv2.arrowedLine(img_show, p0, p1, color, 2)
            # store the FILTERED OF vectors
            flow_vec.append([p1, p1[0]-p0[0]])
            ii = ii + 1

        # calculate the average horizontal OF speed for the filtered OF vectors in each panel
        for i in range(10):
            if cnt_seg[i] == 0:
                avg_hor_seg[i] = 0
            else:
                avg_hor_seg[i] = sum_hor_seg[i] / cnt_seg[i]
        # calculate the average time-to-transit for each panel using the averaged horizontal OF
        for i in range(10):
            if cnt_seg[i] == 0:
                avg_tau_seg[i] = 0
            else:
                if avg_hor_seg[i] == 0:
                    avg_tau_seg[i] = np.inf
                else:
                    avg_tau_seg[i] = (w * (i + 1) / 10 - w / 20 - w / 2) / (avg_hor_seg[i])

        # write the statistics for filtered optical flow into the log
        f.write(str(cnt_seg) + ', ')
        f.write(str(avg_tau_seg) + ', ')

        # initiate visualization
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(9):
            cv2.line(mask_fig, (int(w*(i+1)/10),0), (int(w*(i+1)/10), h), (255, 255, 255), 1)
        # averaged time-to-transit
        cv2.putText(mask_fig, 'avg_ntau', (int(w/2-120), int(h / 3 - 50)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # count of OF vectors in the panel
        cv2.putText(mask_fig, 'pts_cnt', (int(w/2-100), int(h*2 / 3) - 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # If the direction of the horizontal OF velocity is consistent with the motion direction of the camera,
        # draw the OF vector in green; else, draw the OF vector in red (suggesting errors).
        for i in range(10):
            if float(avg_seg[i]) > 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            # plot the averaged time-to-transit and number of OF vectors in the panel
            cv2.putText(mask_fig, '{0:.1f}'.format(avg_tau_seg[i]), (int(w*(i+0.2)/10), int(h / 3)), font, 1,
                        color, 2, cv2.LINE_AA)
            cv2.putText(mask_fig, str(cnt_seg[i]), (int(w * (i + 0.2) / 10), int(h*2/3)), font, 1, color, 2,
                        cv2.LINE_AA)
        # also output the OF statistics in the console
        print(avg_seg)

        # calculate the average tau for the left and right of the frame
        # OF in central panels are discarded as they tend to be noisy
        panel_left = avg_tau_seg[1:4]
        panel_left.sort()
        panel_right = avg_tau_seg[6:9]
        panel_right.sort()
        nzero_left = [0,0,0]
        nzero_right = [0,0,0]

        # examine the number of panels with non-zero OF vectors
        for i in range(3):
            if cnt_seg[i+1] != 0:
                nzero_left[i] = 1
            if cnt_seg[8-i] != 0:
                nzero_right[i] = 1

        if sum(nzero_left) != 0:
            avg_left = abs(sum(avg_seg[1:4]) / sum(nzero_left))
        else:
            avg_left = 0
        if sum(nzero_right) != 0:
            avg_right = abs(sum(avg_seg[6:9]) / sum(nzero_right))
        else:
            avg_right = 0

        left_ct = avg_left
        right_ct = avg_right
        control = k*(left_ct - right_ct)
        if control > 0:
            color = (0, 255, control * 10)
        else:
            color = (-control * 10, 255, 100)
        if control > 0:
            ctstr = 'L'
            print('Left')
        elif control < 0:
            ctstr = 'R'
            print('Right')
        else:
            ctstr = 'N'
            print('normal')

        f.write(str(control) + ',')
        f.write('\n')

        # visualize the magnitude and direction of the steering signal
        cv2.rectangle(mask_fig, (int(w/2), int(h*5/6)), (int(w/2+control/100), int(h*5/6 + 25)), color, -1)
        cv2.putText(mask_fig, '{0:.1f}'.format(control/1000)+ctstr, (int(w / 2 - 30), int(h * 5 / 6 - 25)), font, 1, color, 2, cv2.LINE_AA)

        img = cv2.add(img2, mask_fig)
        img = cv2.add(img, img_show)
        vout.write(img)
        img1 = img2

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', 1280, 720)
        cv2.imshow('Image', img)
        k = cv2.waitKey(10) & 0xff

        print(frame_no)

except:
    traceback.print_exc()
    print('Catched')
    # print(e)
    # cv2.destroyAllWindows()
    vout.release()
    cap.release()
    f.write('\n')
    f.close()
#
# --------------------------------------------------------------------------------------------------------------------