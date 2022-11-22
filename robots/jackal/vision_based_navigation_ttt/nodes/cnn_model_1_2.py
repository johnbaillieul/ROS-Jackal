#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
import rospy
from tensorflow import keras
import openpyxl
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tkinter import W
from sensor_msgs.msg import LaserScan
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt.msg import TauComputation
from cv_bridge import CvBridgeError, CvBridge
from sensor_msgs.msg import Image
import pandas as pd
from xlsxwriter import Workbook
import tensorflow as tf

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
	###########################################

##############################################################################################

# Visual representation of the ROIs with the average TTT values
def draw_image_segmentation(curr_image, tau_el, tau_er, tau_l, tau_r, tau_c):
    color_image = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
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

class train():
    def __init__(self):
        pass
    def train_(self):
        np.random.seed(0)

        X = []
        y = []

        path_tau = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/tau_values/tau_value"   
        path_images = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/images/"
        images_in_folder = [f for f in listdir(path_images)]

        for name in images_in_folder :
            if (name.endswith(".png")):
                try:
                    # load the image
                    img = cv2.imread(path_images + name,0)
                    img = cv2.resize(img,(250,250))
                    image_as_array = np.array(img)
                    # add our image to the dataset
                    X.append(image_as_array)
                    # retrive the direction from the filename
                    ps = openpyxl.load_workbook(path_tau + str(name.split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    y.append(tau_values)
                except Exception as inst:
                    print(name)
                    print(inst)

        X = np.asarray(X)
        print(X.shape)
        y = np.asarray(y)

        # # split for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(X_train.shape)
        # scale the data
        X_train = X_train.reshape(X_train.shape[0],250,250,1)
        # X_train = X_train/255.0
        X_test = X_test.reshape(X_test.shape[0],250,250,1)
        # X_test = X_test/255.0
        print(X_train.shape)
        model = keras.Sequential([keras.layers.Conv2D(64,(3,3),activation='relu',input_shape= (250,250,1)),
                                keras.layers.MaxPooling2D(2,2),
                                keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                keras.layers.MaxPooling2D(2,2),
                                keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                keras.layers.MaxPooling2D(2,2),
                                keras.layers.Flatten(),
                                keras.layers.Dense(units = 128,activation = 'relu',kernel_initializer='he_uniform'),
                                keras.layers.Dropout(0.25),
                                keras.layers.Dense(units = 225,activation = 'relu',kernel_initializer='he_uniform'),
                                keras.layers.Dropout(0.25),
                                keras.layers.Dense(5,kernel_initializer='he_uniform')
                                ])
        model.compile(optimizer = 'adam', loss = 'mae', metrics = 'accuracy')
        model.summary()

        #train the model
        model.fit(X_train,y_train,epochs = 100)
        print(model.predict(X_test)[8])
        test_loss, test_acc = model.evaluate(X_test, y_test)

        print('test loss: {}, test accuracy: {}'.format(test_loss, test_acc) )

        #evaluate one example
        y_pred = model.predict(X_test)
        y_pred_classes = y_pred
        random_idx = np.random.choice(len(X_test))
        x_sample = X_test[random_idx]
        y_sample_true = y_test[random_idx]
        y_sample_pred_class = y_pred_classes[random_idx]
        print('true',y_sample_true, 'pred', y_sample_pred_class)

        with open('model_pkl_1', 'wb') as files: pickle.dump(model, files)

class inference():
    def __init__(self):
        pass

    def extract_model(self):
        path_images = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/images/"
        name = '1.png'
        img = cv2.imread(path_images + name,0)
        img = cv2.resize(img,(250,250))
        # cv2.imshow('img',img)
        # cv2.waitKey(0) 
        img = np.asarray(img)
        # print('sh',img.shape[0])
        img = tf.expand_dims(img, 0)
        # img = img.reshape(1,250,250,1)
        # print(image_as_array)
        pickled_model = pickle.load(open('model_pkl_1.pkl', 'rb'))
        # with open('model_pkl_1' , 'rb') as f: lr = pickle.load(f)
        # print(lr)
        tau_pred = pickled_model.predict([img])
        # print(tau_pred)
        path_tau = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/tau_values/tau_value"   
        ps = openpyxl.load_workbook(path_tau + str(name.split('.')[0]) + '.xlsx')
        sheet = ps['Sheet1']
        tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
        print('tau_pred', tau_pred, 'tau_val',tau_values)

class calc_tau():
    def __init__(self):
        # Tau Publisher
        self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)
        # Raw Image Subscriber
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        # Initialize Image acquisition
        self.bridge = CvBridge()
        # self.get_variables()
        self.curr_image = None
        self.model = pickle.load(open('model_pkl_1.pkl', 'rb'))

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
        if self.curr_image is not None:
            curr_image = self.curr_image
            img = cv2.resize(curr_image,(250,250))
            # cv2.imshow('img',img)
            # cv2.waitKey(0) 
            img = np.asarray(img)
            # print('sh',img.shape[0])
            img = img.reshape(1,250,250,1)
            # print(image_as_array)
        
            # with open('model_pkl_1' , 'rb') as f: lr = pickle.load(f)
            # print(lr)
            tau_pred = self.model.predict([img])
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

            # Draw the ROIs with their TTT values
            draw_image_segmentation(curr_image, tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2])



if __name__ == '__main__':
    rospy.init_node('cnn_from_lidar', anonymous=True)
    tau = calc_tau()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        # tr = train()
        # tr.train_()
        # inf = inference()
        # inf.extract_model()
        tau.get_tau_values()
        r.sleep()
    

