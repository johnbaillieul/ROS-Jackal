#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir
# from sklearn.metrics import confusion_matrix
import rospy
import tensorflow as tf
from tensorflow import keras
from tkinter import W
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt.msg import TauComputation
from cv_bridge import CvBridgeError, CvBridge
from sensor_msgs.msg import Image

############################################################################################
# Initialization of the variables for setting the limits of the ROIs

# Definition of the limits for the ROIs
def set_limit(img_width, img_height):
    
    ########## IMPORTANT PARAMETERS: ##########
	# Extreme left and extreme right
	global x_init_el
	global y_init_el
	global x_end_el
	global y_end_el
	x_init_el = 0
	y_init_el = 0
	x_end_el = int(3 * img_width / 12)
	y_end_el = int(11 * img_height / 12)

	global x_init_er
	global y_init_er
	global x_end_er
	global y_end_er
	x_init_er = int(9 * img_width / 12)
	y_init_er = 0
	x_end_er = int(img_width)
	y_end_er = int(11 * img_height / 12)

	# Left and right
	global x_init_l
	global y_init_l
	global x_end_l
	global y_end_l
	x_init_l = int(3 * img_width / 12)
	y_init_l = int(1 * img_height / 12)
	x_end_l = int(5 * img_width / 12)
	y_end_l = int(9.5 * img_height / 12)

	global x_init_r
	global y_init_r
	global x_end_r
	global y_end_r
	x_init_r = int(7 * img_width / 12)
	y_init_r = int(1 * img_height / 12)
	x_end_r = int(9 * img_width / 12)
	y_end_r = int(9.5 * img_height / 12)
    
    # Centre
	global x_init_c
	global y_init_c
	global x_end_c
	global y_end_c
	x_init_c = int(5.5 * img_width / 12)
	y_init_c = int(2.5 * img_height / 12)
	x_end_c = int(6.5 * img_width / 12)
	y_end_c = int(7.5 * img_height / 12)

##############################################################################################

# Visual representation of the ROIs with the average TTT values
def draw_image_segmentation(curr_image, taup_el, taup_er, taup_l, taup_r, taup_c):
    color_image = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
    color_blue = [255, 225, 0]  
    color_green = [0, 255, 0]
    color_red = [0, 0, 255]
    linewidth = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extreme left and extreme right
    cv2.rectangle(color_image, (x_init_el, y_init_el), (x_end_el, y_end_el), color_blue, linewidth)
    cv2.rectangle(color_image, (x_init_er, y_init_er), (x_end_er, y_end_er), color_blue, linewidth)
    cv2.putText(color_image, str(round(taup_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/3)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 3)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Left and right
    cv2.rectangle(color_image, (x_init_l, y_init_l), (x_end_l, y_end_l), color_green, linewidth)
    cv2.rectangle(color_image, (x_init_r, y_init_r), (x_end_r, y_end_r), color_green, linewidth)
    cv2.putText(color_image, str(round(taup_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(taup_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 3)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation', (600, 600))
    cv2.imshow('ROIs Representation', color_image)
    cv2.waitKey(10)
#######################################################################################################################

class calc_tau():
    def __init__(self):
        # Tau Publisher
        self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)
        # Raw Image Subscriber
        self.image_sub_name = "/camera/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        # Initialize Image acquisition
        self.bridge = CvBridge()
        # self.get_variables()
        self.curr_image = None
        self.prev_image = None
        # make predictions on test sets
        
        # self.model = tf.keras.models.load_model("model_5_without_v_with_flag_img_size_200_epochs_100.h5", compile=False)
        # self.model.compile(optimizer = 'adam', loss = 'mae', metrics = ['MeanSquaredError']) #Paste it here
        self.model = tf.lite.Interpreter(model_path= "model_5_without_v_without_flag_old_data_img_size_200_model.tflite")
        self.model.allocate_tensors()
        self.input_index = self.model.get_input_details()[0]["index"]
        self.output_index = self.model.get_output_details()[0]["index"]

        self.vel = 1
    
    def callback_img(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        # Get time stamp
        self.secs = data.header.stamp.secs
        self.nsecs = data.header.stamp.nsecs
        self.width = data.width
        self.height = data.height

    def get_tau_values(self):
        img_size = 250
        if self.curr_image is not None:
            if self.prev_image is not None:
                print("here")
                curr_image = self.curr_image

                img_1 = cv2.resize(self.prev_image,(img_size,img_size))
                image_as_array_1 = np.array(img_1)
                image_as_array_1 = image_as_array_1.reshape(img_size,img_size,1)

                img_2 = cv2.resize(self.curr_image,(img_size,img_size))
                image_as_array_2 = np.array(img_2)
                image_as_array_2 = image_as_array_2.reshape(img_size,img_size,1)
                # print('2',image_as_array_2.shape)
                # print(self.total_imgs)
                img = np.stack([img_1, img_2], 2)
                img = tf.expand_dims(img, 0)
                # img = img.reshape(1,250,250,1)
                img = np.asarray(img)
                print('img',img.shape)
                # print(img.shape)
                
                vel = np.asarray([self.vel])
                print('v',vel.shape)
                # with open('model_pkl_1' , 'rb') as f: lr = pickle.load(f)
                # print(lr)

                inf_image = img.astype(np.float32)
                self.model.set_tensor(self.input_index, inf_image)
                self.model.invoke()
                tau_pred = self.model.get_tensor(self.output_index)

                # tau_pred = self.model.predict({"input_1": img, "input_2": vel})
                set_limit(self.width, self.height)

                # Publish Tau values data to rostopic
                # Creation of TauValues.msg
                msg = TauComputation()
                msg.header.stamp.secs =  self.secs
                msg.header.stamp.nsecs =  self.nsecs
                msg.height = self.height
                msg.width = self.width

                msg.tau_el = tau_pred[0][0]
                msg.tau_er = tau_pred[0][4]
                msg.tau_l = tau_pred[0][1]
                msg.tau_r = tau_pred[0][3]
                msg.tau_c = tau_pred[0][2]
                self.tau_values.publish(msg)
                self.prev_image = self.curr_image
                
                draw_image_segmentation(curr_image, tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2])
            
            else:  
                self.prev_image = self.curr_image

    def get_tau_values_without_v(self):
        if self.curr_image is not None:
            if self.prev_image is not None:
                print("here")
                img_size = 250
                curr_image = self.curr_image

                img_1 = cv2.resize(self.prev_image,(img_size,img_size))
                image_as_array_1 = np.array(img_1)
                image_as_array_1 = image_as_array_1.reshape(img_size,img_size,1)

                img_2 = cv2.resize(self.curr_image,(img_size,img_size))
                image_as_array_2 = np.array(img_2)
                image_as_array_2 = image_as_array_2.reshape(img_size,img_size,1)
               
                img = np.stack([img_1, img_2], 2)
                img = tf.expand_dims(img, 0)
               
                img = np.asarray(img)
               
                vel = 1
                vel = np.asarray([vel])
              
                # with open('model_pkl_1' , 'rb') as f: lr = pickle.load(f)
                # print(lr)
                inf_image = img.astype(np.float32)
                self.model.set_tensor(self.input_index, inf_image)
                self.model.invoke()
                tau_pred = self.model.get_tensor(self.output_index)
                # tau_pred = self.model.predict({"input_1": img})
                set_limit(self.width, self.height)

                # Publish Tau values data to rostopic
                # Creation of TauValues.msg
                msg = TauComputation()
                msg.header.stamp.secs =  self.secs
                msg.header.stamp.nsecs =  self.nsecs
                msg.height = self.height
                msg.width = self.width

                msg.tau_el = tau_pred[0][0]/vel
                msg.tau_er = tau_pred[0][4]/vel
                msg.tau_l = tau_pred[0][1]/vel
                msg.tau_r = tau_pred[0][3]/vel
                msg.tau_c = tau_pred[0][2]/vel
                self.tau_values.publish(msg)
                self.prev_image = self.curr_image
                
                draw_image_segmentation(curr_image, tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2])
            
            else:  
                self.prev_image = self.curr_image

if __name__ == '__main__':
    # tau_computation_from_cnn()
    rospy.init_node('cnn_on_jackal', anonymous=True)
    tau = calc_tau()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        tau.get_tau_values()
        r.sleep()
    

