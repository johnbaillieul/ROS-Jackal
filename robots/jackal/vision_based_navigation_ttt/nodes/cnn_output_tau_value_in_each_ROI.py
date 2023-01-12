#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir
import rospy
import tensorflow as tf
from tensorflow import keras
import openpyxl
from PIL import Image
import matplotlib.pyplot as plt
import os
from tkinter import W
from sensor_msgs.msg import LaserScan, Image
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt.msg import TauComputation
import pandas as pd
from xlsxwriter import Workbook

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

# Visual representation of the ROIs with the average TTT values
def draw_image_segmentation(curr_image, taup_el, taup_er, taup_l, taup_r, taup_c,tau_el, tau_er, tau_l, tau_r, tau_c):
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
    cv2.putText(color_image, str(round(taup_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/3)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 3)),
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
    cv2.putText(color_image, str(round(taup_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(tau_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 2)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 3)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation', (600, 600))
    cv2.imshow('ROIs Representation', color_image)
    cv2.waitKey(10)

class Train():
    def __init__(self):
        pass
    def train_(modelself):
        np.random.seed(0)

        X = []
        y = []
        velocity = []

        # path = os.environ["HOME"]+"/catkin_ws/src/"
        path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/"
        path_tau =  path + "vision_based_navigation_ttt/tau_values_no_flag/tau_value"   
        path_folder = path + "vision_based_navigation_ttt/training_images/"

        folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
        for folder in folders:

            # print('fol',folder)
            path_images = path + "vision_based_navigation_ttt/training_images/" + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]
            img_size = 250
            for idx in range(len(images_in_folder)-1):
                try:
                    # load the image
                    img_1 = cv2.imread(path_images + images_in_folder[idx],0)
                    img_1 = cv2.resize(img_1,(img_size,img_size))

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1],0)
                    img_2 = cv2.resize(img_2,(img_size,img_size))
                    
                    img = np.stack([img_1, img_2], 2)
                    
                    # add image to the dataset
                    X.append(img)

                    # get velocity
                    vel = float(folder.split('_')[4])
                    velocity.append([vel])

                    # retrive distances
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx+1].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    y.append(np.asarray(tau_values)/vel)

                except Exception as inst:
                    print(idx)
                    print(inst)
         
        X = np.asarray(X)
        y = np.asarray(y)
        velocity = np.asarray(velocity)
        
        ind = np.arange(len(X))
        np.random.shuffle(ind)

        # # split the data in 60:20:20 for train:valid:test dataset
        train_size = 0.6
        valid_size = 0.2

        train_index = int(len(ind)*train_size)
        valid_index = int(len(ind)*valid_size)

        X_train = X[ind[0:train_index]]
        X_valid = X[ind[train_index:train_index+valid_index]]
        X_test = X[ind[train_index+valid_index:]]

        y_train = y[ind[0:train_index]]
        y_valid = y[ind[train_index:train_index+valid_index]]
        y_test = y[ind[train_index+valid_index:]]

        v_train = velocity[ind[0:train_index]]
        v_valid = velocity[ind[train_index:train_index+valid_index]]
        v_test = velocity[ind[train_index+valid_index:]]

        # # Convolutional Neural Network
        input_1 = keras.layers.Input(shape=(img_size,img_size,2))
        conv1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(input_1)
        pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(pool1)
        pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(pool2)
        pool3 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(pool3)
        input_2 = keras.layers.Input(shape=(1))

        # merge input models
        merge = keras.layers.concatenate([flat, input_2])  

        hidden1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(merge)
        drop_1 = keras.layers.Dropout(0.25)
        hidden2 = keras.layers.Dense(225, activation='relu',kernel_initializer='he_uniform')(hidden1)
        drop_1 = keras.layers.Dropout(0.25)(hidden2)
        output = keras.layers.Dense(5,kernel_initializer='he_uniform')(drop_1)

        model = keras.models.Model(inputs=[input_1, input_2], outputs=output)

        # # summarize layers
        print(model.summary())
        # # plot graph
        model_name = "model_with_shape_info"
        keras.utils.plot_model(model, model_name + ".png")

        model.compile(optimizer = 'adam', loss = 'mae', metrics = 'accuracy')
      
        ## train the model
        model.fit({"input_1": X_train, "input_2": v_train}, y_train, batch_size=64, epochs = 100,  validation_data=({"input_1": X_valid, "input_2": v_valid}, y_valid))
        # Save the best performing model to file
        model.save(model_name + ".h5")

        # print(model.predict(X_test)[8])
        test_loss, test_acc = model.evaluate({"input_1": X_test, "input_2": v_test}, y_test)
        print('test loss: {}, test accuracy: {}'.format(test_loss, test_acc) )

class Calc_Tau():
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
        self.prev_image = None
        self.model = tf.lite.Interpreter(model_path= "model_5_without_v_without_flag_old_data_img_size_200_model.tflite")
        self.model.allocate_tensors()
        self.input_index = self.model.get_input_details()[0]["index"]
        self.output_index = self.model.get_output_details()[0]["index"]
        # self.model = tf.keras.models.load_model("model_5_without_v_with_flag_img_size_200_epochs_100.h5", compile=False)
        # self.model.compile(optimizer = 'adam', loss = 'mae', metrics = ['MeanSquaredError', 'mean_absolute_error']) #Paste it here
        self.vel = 1
        # Lidar Subscriber
        self.sub = rospy.Subscriber('/front/scan', LaserScan, self.callback)
        self.ranges = None
        self.increments = None
        self.linear_x_vel = 1
        self.angle_min = None
        self.angle_max = None

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

    def get_tau_values_with_v(self):
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
                vel = 1
                vel = np.asarray([vel])
                print('v',vel.shape)
                # with open('model_pkl_1' , 'rb') as f: lr = pickle.load(f)
                # print(lr)
                tau_pred = self.model.predict({"input_1": img, "input_2": vel})
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
            
                draw_image_segmentation(curr_image, tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2], self.tau_el, self.tau_er, self.tau_l, self.tau_r, self.tau_c)
            else:  
                self.prev_image = self.curr_image

    def get_tau_values_without_v(self):
        if self.curr_image is not None:
            if self.prev_image is not None:
                print("here")
                img_size = 200
                curr_image = self.curr_image

                img_1 = cv2.resize(self.prev_image,(img_size,img_size))
                img_2 = cv2.resize(self.curr_image,(img_size,img_size))
               
                img = np.stack([img_1, img_2], 2)
                img = tf.expand_dims(img, 0)
                img = np.asarray(img)
             
                vel = np.asarray([self.vel])
              
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
                
                draw_image_segmentation(curr_image, tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2], self.tau_el, self.tau_er, self.tau_l, self.tau_r, self.tau_c)
            else:  
                self.prev_image = self.curr_image

    def callback_lidar(self, msg):
        start_ind  = 230 #0
        end_ind = 488 #len(msg.ranges) - 1   #488 
        self.angle_min = msg.angle_min + start_ind * msg.angle_increment
        self.angle_max = msg.angle_min + end_ind * msg.angle_increment
        self.increments = msg.angle_increment
        self.ranges = msg.ranges[230:489]
        if self.ranges is not None:
            theta_rd = np.arange(self.angle_min,self.angle_max + self.increments, self.increments, dtype=float) # generated within the half-open interval [start, stop).
            theta_deg = theta_rd * (180/np.pi)
            ranges = self.ranges
            tau_val = np.array([])
            for i in range(len(ranges)):
                tau_val = np.append(tau_val,abs(ranges[i]*np.cos(theta_deg[i])))
            self.tau_val = tau_val/self.linear_x_vel
        
            # Extreme left
            count_inf_el = 0
            count_el = 0
            tau_val_el = self.tau_val[204:259]
            tau_el = 0
            for i in range(len(tau_val_el)):
                if tau_val_el[i] == np.inf: 
                    count_inf_el += 1
                    if count_inf_el > int(len(tau_val_el)/2):
                        tau_el = np.inf
                        break
                else:
                    tau_el += tau_val_el[i]
                    count_el += 1
            if count_el ==0:
                tau_el = np.inf
            else:
                tau_el = tau_el / count_el

            # Extreme right
            count_inf_er = 0
            count_er = 0
            tau_val_er = tau_val[0:56]
            tau_er = 0       
            for i in range(len(tau_val_er)):
                if tau_val_er[i] == np.inf: 
                    count_inf_er += 1
                    if count_inf_er > int(len(tau_val_er)/2):
                        tau_er = np.inf
                        break
                else: 
                    tau_er += tau_val_er[i]
                    count_er += 1
            if count_er == 0:
                tau_er = np.inf
            else:
                tau_er = tau_er / count_er

            # left
            count_inf_l = 0
            count_l = 0
            tau_val_l = tau_val[154:205]
            tau_l = 0
            for i in range(len(tau_val_l)):
                if tau_val_l[i] == np.inf: 
                    count_inf_l += 1
                    if count_inf_l > int(len(tau_val_l)/2):
                        tau_l = np.inf
                        break
                else: 
                    tau_l += tau_val_l[i]
                    count_l += 1
            if count_l == 0:
                tau_l = np.inf
            else:
                tau_l = tau_l / count_l

            # Right
            count_inf_r = 0
            count_r = 0
            tau_val_r = tau_val[55:103]
            tau_r = 0
            for i in range(len(tau_val_r)):
                if tau_val_r[i] == np.inf: 
                    count_inf_r += 1
                    if count_inf_r > int(len(tau_val_r)/2):
                        tau_r = np.inf
                        break
                else: 
                    tau_r += tau_val_r[i]
                    count_r += 1
            if count_r == 0:
                tau_r = np.inf
            else:
                tau_r = tau_r / count_r

            # Centre
            count_inf_c = 0
            count_c = 0
            tau_val_c = tau_val[117:144]
            tau_c = 0
            for i in range(len(tau_val_c)):
                if tau_val_c[i] == np.inf: 
                    count_inf_c+=1
                    if count_inf_c > int(len(tau_val_c)/2):
                        tau_c = np.inf
                        break
                else: 
                    tau_c += tau_val_c[i]
                    count_c +=1
            if count_c == 0:
                tau_c = np.inf
            else:
                tau_c = tau_c / count_c 
            self.tau_el = tau_el
            self.tau_l = tau_l
            self.tau_c = tau_c
            self.tau_r = tau_r
            self.tau_er = tau_er

if __name__ == '__main__':
    # tau_computation_from_cnn()
    rospy.init_node('cnn_from_lidar', anonymous=True)
    tau = Calc_Tau()
    r = rospy.Rate(10)
    # tr = train()
    # tr.train_()
    while not rospy.is_shutdown():
        # tr = train()
        # tr.train_()
        tau.get_tau_values_without_v()
        r.sleep()
    

