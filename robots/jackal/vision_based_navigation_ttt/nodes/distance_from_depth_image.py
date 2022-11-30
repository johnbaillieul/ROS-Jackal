#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from vision_based_navigation_ttt.msg import OpticalFlow
from cv_bridge import CvBridgeError, CvBridge
import cv2
import sys
import numpy as np

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
	y_end_er = int( img_height )

	# Left and right
	global x_init_l
	global y_init_l
	global x_end_l
	global y_end_l
	x_init_l = int(3 * img_width / 12)
	y_init_l = 0
	x_end_l = int(5 * img_width / 12)
	y_end_l = int(img_height)

	global x_init_r
	global y_init_r
	global x_end_r
	global y_end_r
	x_init_r = int(7 * img_width / 12)
	y_init_r = 0
	x_end_r = int(9 * img_width / 12)
	y_end_r = int(img_height)

      # Centre
	global x_init_c
	global y_init_c
	global x_end_c
	global y_end_c
	x_init_c = int(5 * img_width / 12)
	y_init_c = 0
	x_end_c = int(7 * img_width / 12)
	y_end_c = int(img_height)
	###########################################


class DepthImage:
    def __init__(self):
        self.image_sub_name = "/realsense/depth/image_rect_raw" 
        # depth Image Subscriber 
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback)
        self.bridge = CvBridge()
        # self.image = rospy.Subscriber("/realsense/color/image_raw", Image, self.callback__)

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
        set_limit(msg.width, msg.height)
        
        
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