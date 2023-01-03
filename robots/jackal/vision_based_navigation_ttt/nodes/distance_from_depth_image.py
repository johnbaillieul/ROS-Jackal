#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Image as msg_Image
from vision_based_navigation_ttt.msg import OpticalFlow
from cv_bridge import CvBridgeError, CvBridge
import cv2
import sys
import numpy as np
# import pyrealsense2 as rs2
# if (not hasattr(rs2, 'intrinsics')):
#     import pyrealsense2.pyrealsense2 as rs2
from sensor_msgs.msg import CameraInfo

# Visual representation of the ROIs with the average TTT values
def draw_image_segmentation(curr_image, tau_el, tau_er, tau_l, tau_r, tau_c):

    color_image = curr_image
    color_blue = [255, 225, 0]  
    color_green = [0, 255, 0]
    color_red = [0, 0, 255]
    linewidth = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extreme left and extreme right
    cv2.rectangle(color_image, (x_init_el, y_init_el), (x_end_el, y_end_el), color_blue, linewidth)
    cv2.rectangle(color_image, (x_init_er, y_init_er), (x_end_er, y_end_er), color_blue, linewidth)
    cv2.putText(color_image, str(round(tau_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/2)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 2)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Left and right
    cv2.rectangle(color_image, (x_init_l, y_init_l), (x_end_l, y_end_l), color_green, linewidth)
    cv2.rectangle(color_image, (x_init_r, y_init_r), (x_end_r, y_end_r), color_green, linewidth)
    cv2.putText(color_image, str(round(tau_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(tau_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 2)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation', (600, 600))
    cv2.imshow('ROIs Representation', color_image)
    cv2.waitKey(10)

#######################################################################################################################

def set_limit(img_width, img_height):
    # Extreme left and extreme right
	global x_init_el
	global y_init_el
	global x_end_el
	global y_end_el
	x_init_el = 0
	y_init_el = 0
	x_end_el = int(3 * img_width / 12)
	y_end_el = int(img_height)

	global x_init_er
	global y_init_er
	global x_end_er
	global y_end_er
	x_init_er = int(9 * img_width / 12)
	y_init_er = 0
	x_end_er = int(img_width)
	y_end_er = int(img_height)

	# Left and right
	global x_init_l
	global y_init_l
	global x_end_l
	global y_end_l
	x_init_l = int(3 * img_width / 12)
	y_init_l = int((img_width / 15))
	x_end_l = int(5 * img_width / 12)
	y_end_l = int(img_height- (img_width / 15))

	global x_init_r
	global y_init_r
	global x_end_r
	global y_end_r
	x_init_r = int(7 * img_width / 12)
	y_init_r = int(img_width / 15)
	x_end_r = int(9 * img_width / 12)
	y_end_r = int(img_height- (img_width / 15))

      # Centre
	global x_init_c
	global y_init_c
	global x_end_c
	global y_end_c
	x_init_c = int(5 * img_width / 12)
	y_init_c = int(img_width / 11)
	x_end_c = int(7 * img_width / 12)
	y_end_c = int(img_height- (img_width / 11))
	###########################################

class DepthImage:
    def __init__(self):
        self.image_sub_name = "/realsense/color/image_raw" 
        self.depth_topic = "/realsense/depth/image_rect_raw" 
        # depth Image Subscriber 
        self.depth_image_sub = rospy.Subscriber(self.depth_topic, msg_Image, self.imageDepthCallback)
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.img_callback)
        # self.sub_info = rospy.Subscriber(self.depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        self.bridge = CvBridge()
        self.curr_image = None
        # self.image = rospy.Subscriber("/realsense/color/image_raw", Image, self.callback__)

    def imageDepthCallback(self, data):
        
        curr_image = self.curr_image
        cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        # print('cv_image_size',np.size(cv_image))

        set_limit(data.width, data.height)

        roi_el = cv_image[int(y_init_el):int(y_end_el), int(x_init_el):int(x_end_el)].reshape(1,-1)
        roi_er = cv_image[int(y_init_er):int(y_end_er), int(x_init_er):int(x_end_er)].reshape(1,-1)
        roi_l = cv_image[int(y_init_l):int(y_end_l), int(x_init_l):int(x_end_l)].reshape(1,-1)
        roi_r = cv_image[int(y_init_r):int(y_end_r), int(x_init_r):int(x_end_r)].reshape(1,-1)
        roi_c = cv_image[int(y_init_c):int(y_end_c), int(x_init_c):int(x_end_c)].reshape(1,-1)
        # print('type',np.shape(roi_el))
        # print("el",roi_el)
        # print("er",roi_er)
        # print("l",roi_l)
        # print("r",roi_r)
        # print("c",roi_c)
     
        # print('r', roi_el)
        # roi_el_median = np.median(roi_el)
        # # print('rfff', roi_el_median)
        # indices = np.where(not np.isnan(roi_er[0][:]))
        # diffs = diffs[indices]
        # print('sss',roi_er[indices])
        # roi_er_median = np.median(roi_er[not np.isnan(roi_er)])

        # roi_l_median = np.median(roi_l[])
        # roi_r_median = np.median(roi_r[])
        # roi_c_median = np.median(roi_c[]) 
        # cv2.imshow("depth image", roi_el)
        # cv2.waitKey()
        # cv2.imshow("depth image", roi_l)
        # cv2.waitKey()
        # cv2.imshow("depth image", roi_c)
        # cv2.waitKey()
        # cv2.imshow("depth image", roi_r)
        # cv2.waitKey()
        # cv2.imshow("depth image", roi_er)
        # cv2.waitKey()
        limit = 100000
        tau_el = roi_el[np.logical_not(np.isnan(roi_el))]
        print('el',np.size(tau_el))
        if np.size(tau_el) > limit:
            tau_el_median = np.mean(tau_el)
        else:
            tau_el_median = -1

        tau_er = roi_er[np.logical_not(np.isnan(roi_er))]
        print(np.size(tau_er))
        if np.size(tau_er) > limit:
            tau_er_median = np.mean(tau_er)
        else:
            tau_er_median = -1

        tau_l = roi_l[np.logical_not(np.isnan(roi_l))]
        print(np.size(tau_l))
        if np.size(tau_l) > limit:
            tau_l_median = np.mean(tau_l)
        else:
            tau_l_median = -1

        tau_r = roi_r[np.logical_not(np.isnan(roi_r))]
        print(np.size(tau_r))
        if np.size(tau_r) > limit:
            tau_r_median = np.mean(tau_r)
        else:
            tau_r_median = -1

        tau_c = roi_c[np.logical_not(np.isnan(roi_c))]
        print(tau_c)
        print(np.size(tau_c))
        if np.size(tau_c) > limit:
            tau_c_median = np.mean(tau_c)
        else:
            tau_c_median = -1
        
        
        # print('tau_el',tau_el)
        # print('sahpe',np.shape(tau_el))
        # if tau_el
        # count_inf_el = 0
        # count_el = 0
        # tau_val_el = roi_el
        # tau_el = 0
        # for i in range(len(tau_val_el)):
        #     # print('r',np.shape(roi_el))
        #     print('1,2', tau_val_el[0][i])
        #     # print('ola', np.float32('nan'))
        #     # print('ola', type(np.float32('nan')))
        #     # print('ola', type(tau_val_el[0][i]))
        #     if np.isnan(tau_val_el[0][i]): 
        #         print('1')
        #         count_inf_el += 1
        #         if count_inf_el > int(len(tau_val_el)/2):
        #             tau_el = -1
        #             break
        #     else:
        #         print('2')
        #         tau_el += tau_val_el[0][i]
        #         count_el += 1
        # if count_el ==0:
        #     tau_el = -1
        # else:
        #     tau_el = tau_el / count_el

        # print("debug2")
        # Extreme right
        # count_inf_er = 0
        # count_er = 0
        # tau_val_er = roi_er
        # tau_er = 0   
        # # print('3,4', tau_val_er[0][i])    
        # for i in range(len(tau_val_er)):
        #     if np.isnan(tau_val_er[0][i]): 
        #         # print('3')
        #         count_inf_er += 1
        #         if count_inf_er > int(len(tau_val_er)/2):
        #             tau_er = -1
        #             break
        #     else:
        #         # print('4') 
        #         tau_er += tau_val_er[0][i]
        #         count_er += 1
        # if count_er == 0:
        #     tau_er = -1
        # else:
        #     tau_er = tau_er / count_er

        # left
        # count_inf_l = 0
        # count_l = 0
        # tau_val_l = roi_l
        # tau_l = 0
        # # print('5,6', tau_val_l[0][i])
        # for i in range(len(tau_val_l)):
        #     if np.isnan(tau_val_l[0][i]):
        #         # print('5') 
        #         count_inf_l += 1
        #         if count_inf_l > int(len(tau_val_l)/2):
        #             tau_l = -1
        #             break
        #     else: 
        #         # print('6')
        #         tau_l += tau_val_l[0][i]
        #         count_l += 1
        # if count_l == 0:
        #     tau_l = -1
        # else:
        #     tau_l = tau_l / count_l

        # # Right
        # count_inf_r = 0
        # count_r = 0
        # tau_val_r = roi_r
        # tau_r = 0
        # # print('7.8', tau_val_r[0][i])
        # for i in range(len(tau_val_r)):
        #     if np.isnan(tau_val_r[0][i]): 
        #         # print('7')
        #         count_inf_r += 1
        #         if count_inf_r > int(len(tau_val_r)/2):
        #             tau_r = -1
        #             break
        #     else: 
        #         # print('8')
        #         tau_r += tau_val_r[0][i]
        #         count_r += 1
        # if count_r == 0:
        #     tau_r = -1
        # else:
        #     tau_r = tau_r / count_r

        # # # print("debug3")
        # # Centre
        # count_inf_c = 0
        # count_c = 0
        # tau_val_c = roi_c
        # tau_c = 0
        # # print('9,10', tau_val_c[0][i])
        # for i in range(len(tau_val_c)):
        #     if np.isnan(tau_val_c[0][i]):
        #         # print('9') 
        #         count_inf_c+=1
        #         if count_inf_c > int(len(tau_val_c)/2):
        #             tau_c = -1
        #             break
        #     else:
        #         # print('10') 
        #         tau_c += tau_val_c[0][i]
        #         count_c +=1

        # if count_c == 0:
        #     tau_c = -1
        # else:
        #     tau_c = tau_c / count_c 
        # print(tau_el, 'oooaaoao')
        # print("debug3")
        # draw_image_segmentation(curr_image, tau_el, tau_er, tau_l, tau_r, tau_c)      
        draw_image_segmentation(curr_image, tau_el_median, tau_er_median, tau_l_median, tau_r_median, tau_c_median)       
# 
        # # except CvBridgeError as e:
        # #     print(e)
        # #     return
        # # except ValueError as e:
        # #     return

    def img_callback(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "passthrough") #"passthrough"
            # cv2.imshow("",curr_image)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)
            return
        # Get time stamp

        self.secs = data.header.stamp.secs
        self.nsecs = data.header.stamp.nsecs
        self.width = data.width
        self.height = data.height


    def callback(self, msg):
        # print(msg.width)
        # print(msg.height)
        # print(msg.data)
        # str = str(msg.data)
        # flatten_depth_img = np.frombuffer(msg.data, dtype=np.uint8)  # shape =(width*height,)
        # print(flatten_depth_img)
        depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)       
        depth_array= depth_array.reshape(msg.height, msg.width) # shape =(width, height)
        # print(depth_array)
        
        # depth_byte_to_array = np.frombuffer(msg.data, np.int16)
        # print(depth_byte_to_array)
        # depth_array = depth_byte_to_array.reshape(msg.width, msg.height)
        # print(msg.data)
        # depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # depth_array = np.array(depth_image, dtype=np.float32)
        
        # print(np.shape(depth_array))
        # print(np.size(msg.data.astype(int)))
        
        # roi_el = depth_array[int(y_init_el):int(y_end_el), int(x_init_el):int(x_end_el)]
        # roi_er= depth_array[int(y_init_er):int(y_end_er), int(x_init_er):int(x_end_er)]
        # roi_l = depth_array[int(y_init_l):int(y_end_l), int(x_init_l):int(x_end_l)]
        # roi_r = depth_array[int(y_init_r):int(y_end_r), int(x_init_r):int(x_end_r)]
        # roi_c = depth_array[int(y_init_c):int(y_end_c), int(x_init_c):int(x_end_c)]
        # print(roi_er_end)
        
        
        # cv2.imshow("depth image", depth_array)
        # cv2.waitKey(5)
        # cv2.imshow("depth image", depth_array[y_init_el:y_end_el, x_init_el:(x_end_el + int(msg.width / 10))])
        # # waiting using waitKey method
        # cv2.waitKey(5)
        
        # roi_er_top = depth_array[int(y_init_er):int(y_end_er/3), int(x_init_er):int(x_end_er)]
        
        # print(roi_er_top)
        
        # roi_er_mid = depth_array[int(y_end_er/3):int(2*y_end_er/3), int(x_init_er):int(x_end_er)]
        roi_er_end = np.median(depth_array[int(2*y_end_er/3):int(3*y_end_er/3), int(x_init_er):int(x_end_er)])
        print(roi_er_end)
        # roi_el_top = depth_array[int(y_init_el):int(y_end_el/3), int(x_init_el):int(x_end_el)]
        # roi_el_mid = depth_array[int(y_end_el/3):int(2*y_end_el/3), int(x_init_el):int(x_end_el)]
        # roi_el_end = depth_array[int(2*y_end_el/3):int(3*y_end_el/3), int(x_init_el):int(x_end_el)]

        # roi_l_top = depth_array[int(y_init_l):int(y_end_l/3), int(x_init_l):int(x_end_l)]
        # roi_l_mid = depth_array[int(y_end_l/3):int(2*y_end_l/3), int(x_init_l):int(x_end_l)]
        # roi_l_end = depth_array[int(2*y_end_l/3):int(3*y_end_l/3), int(x_init_l):int(x_end_l)]

        # roi_r_top = depth_array[int(y_init_r):int(y_end_r/3), int(x_init_r):int(x_end_r)]
        # roi_r_mid = depth_array[int(y_end_r/3):int(2*y_end_r/3), int(x_init_r):int(x_end_r)]
        # roi_r_end = depth_array[int(2*y_end_r/3):int(3*y_end_r/3), int(x_init_r):int(x_end_r)]

        # roi_c_top = depth_array[int(y_init_c):int(y_end_c/3), int(x_init_c):int(x_end_c)]
        # roi_c_mid = depth_array[int(y_end_c/3):int(2*y_end_c/3), int(x_init_c):int(x_end_c)]
        # roi_c_end = depth_array[int(2*y_end_c/3):int(3*y_end_c/3), int(x_init_c):int(x_end_c)]

        
        # cv2.imshow("depth image 1", roi_er_top)
        # cv2.imshow("depth image 2", roi_er_mid )
        # cv2.imshow("depth image 3", roi_er_end)

        # cv2.imshow("depth image 4", roi_el_top)
        # cv2.imshow("depth image 5", roi_el_mid )
        # cv2.imshow("depth image 6", roi_el_end)

        # cv2.imshow("depth image 7", roi_r_top)
        # cv2.imshow("depth image 8", roi_r_mid )
        # cv2.imshow("depth image 9", roi_r_end)

        # cv2.imshow("depth image 10", roi_l_top)
        # cv2.imshow("depth image 11", roi_l_mid )
        # cv2.imshow("depth image 12", roi_l_end)

        # cv2.imshow("depth image 13", roi_c_top)
        # cv2.imshow("depth image 14", roi_c_mid )
        # cv2.imshow("depth image 15", roi_c_end)

        # cv2.waitKey(5000)
     
       
        
        

        
        
        # self.roi_l = curr_image[y_init_l:y_end_l, (x_init_l - int(msg.width / 10)):x_end_l]
        # self.roi_r = curr_image[y_init_r:y_end_r, x_init_r:(x_end_r + int(msg.width /10))]
        # self.roi_c = curr_image[y_init_c:y_end_c, x_init_c:x_end_c]
        # cv2.imshow("depth image", roi_el)
        set_limit(msg.width, msg.height)
       
if __name__ == '__main__':
    rospy.init_node('depth', anonymous=True)
    di = DepthImage()
    r = rospy.Rate(5)
    while not rospy.is_shutdown():
        r.sleep()


### important for later
# from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
# image_sub = Subscriber(
#     "/realsense/color/image_raw",
#     sensor_msgs.msg.Image,
#     queue_size=1
# )

# depth_sub = Subscriber(
#     "/realsense/depth/image_rect_raw",
#     sensor_msgs.msg.Image,
#     queue_size=1
# )

# # Time syncronizer is implimented to make sure that all of the frames match up from all of the topics.
# #ts = TimeSynchronizer([image_sub, depth_sub], 10)
# ts = ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.1)
# ts.registerCallback(self.mycallback)