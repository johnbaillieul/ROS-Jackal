#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow import keras
import openpyxl
from PIL import Image
import os
from tkinter import W
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

#######################################################################################################################

class train():
    def __init__(self):
        pass
    def train_model(self):
        np.random.seed(0)
        width = 150 # img.shape[1] 
        height = 150 # img.shape[0]

        X = []
        y_1 = [] # tau values 
        y_2 = [] # validity 
        velocity = []

        path = os.environ["HOME"]+"/catkin_ws/src/" # change this according to the location of the folder on your device
        path_tau = path + "vision_based_navigation_ttt/tau_values/tau_value"   
        path_folder = path + "vision_based_navigation_ttt/training_images/"

        folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
        for folder in folders:
            path_images = path + "vision_based_navigation_ttt/training_images/" + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]

            for idx in range(len(images_in_folder)-1) : 
                print(images_in_folder[idx])
                try:
                    # load the image
                    img_1 = cv2.imread(path_images + images_in_folder[idx],0)
                    img_1 = cv2.resize(img_1,(150,150))

                    img_el_1 = img_1[0 : int((11/12)*height), int((0/12)*width) : int((2.5/12)*width) ] 
                    img_l_1 = img_1[0 : int((11/12)*height), int((2.5/12)*width) : int((5/12)*width) ] 
                    img_c_1 = img_1[0 : int((11/12)*height), int((4.5/12)*width) : int((7/12)*width) ] 
                    img_r_1 = img_1[0 : int((11/12)*height), int((8.5/12)*width) : int((11/12)*width) ] 
                    img_er_1 = img_1[0 : int((11/12)*height), int((9.4/12)*width) : int((11.9/12)*width) ]

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1],0)
                    img_2 = cv2.resize(img_2,(150,150))

                    img_el_2 = img_2[0 : int((11/12)*height), int((0/12)*width) : int((2.5/12)*width) ] 
                    img_l_2 = img_2[0 : int((11/12)*height), int((2.5/12)*width) : int((5/12)*width) ] 
                    img_c_2 = img_2[0 : int((11/12)*height), int((4.5/12)*width) : int((7/12)*width) ] 
                    img_r_2 = img_2[0 : int((11/12)*height), int((8.5/12)*width) : int((11/12)*width) ] 
                    img_er_2 = img_2[0 : int((11/12)*height), int((9.4/12)*width) : int((11.9/12)*width) ] 
                  
                    img_el = np.stack([img_el_1, img_el_2], 2)
                    img_l = np.stack([img_l_1, img_l_2], 2)
                    img_c = np.stack([img_c_1, img_c_2], 2)
                    img_r = np.stack([img_r_1, img_r_2], 2)
                    img_er = np.stack([img_er_1, img_er_2], 2)
                
                    # add image to the dataset
                    X.append(img_el)
                    X.append(img_l)
                    X.append(img_c)
                    X.append(img_r)
                    X.append(img_er)

                    # get velocity
                    vel = float(folder.split('_')[4])
                    velocity.append([vel])
                    velocity.append([vel])
                    velocity.append([vel])
                    velocity.append([vel])
                    velocity.append([vel])

                    # retrive the tau values from the filename
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx+1].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    #labeled in the following order [el,l,c,r,er]
                    tau_values = [sheet['A1'].value, sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    
                    y_1.append(np.asarray(tau_values[0])/vel)
                    y_1.append(np.asarray(tau_values[1])/vel)
                    y_1.append(np.asarray(tau_values[2])/vel)
                    y_1.append(np.asarray(tau_values[3])/vel)
                    y_1.append(np.asarray(tau_values[4])/vel)

                    for i in tau_values:
                        if i == -1:
                            y_2.append(0)
                        else:
                            y_2.append(1)

                except Exception as inst:
                    print(idx)
                    print(inst)
         
        X = np.asarray(X)
        # print('x',X.shape)
        y_1 = np.asarray(y_1)
        # print(y_1)
        # print('y',y.shape)
        y_2 = np.asarray(y_2)
        # print(y_2)
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

        y_1_train = y_1[ind[0:train_index]]
        y_1_valid = y_1[ind[train_index:train_index+valid_index]]
        y_1_test = y_1[ind[train_index+valid_index:]]

        y_2_train = y_2[ind[0:train_index]]
        y_2_valid = y_2[ind[train_index:train_index+valid_index]]
        y_2_test = y_2[ind[train_index+valid_index:]]

        v_train = velocity[ind[0:train_index]]
        v_valid = velocity[ind[train_index:train_index+valid_index]]
        v_test = velocity[ind[train_index+valid_index:]]

        # # Convolutional Neural Network
        input_1 = keras.layers.Input(shape=(137,31,2))
        conv1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(input_1)
        pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(pool1)
        pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(pool2)
        pool3 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(pool3)
        hidden_1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(flat)
        output_2 = keras.layers.Dense(1, activation= 'sigmoid', name = 'output_2')(hidden_1)
        input_2 = keras.layers.Input(shape=(1))

        # merge input models
        merge = keras.layers.concatenate([flat, input_2])  

        hidden1 = keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(merge)
        
        hidden2 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(hidden1)
        drop_2 = keras.layers.Dropout(0.25)(hidden2)
        output_1 = keras.layers.Dense(1, name = 'output_1')(drop_2)
        

        model = keras.models.Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

        # # summarize layers
        print(model.summary())
        # # plot graph
        keras.utils.plot_model(model, "model_ROI.png", show_shapes=True)
        
        model.compile(optimizer = 'adam', loss = {"output_1" :'mae', "output_2" :'binary_crossentropy'})
        epochs = 100
        ## train the model
        model.fit({"input_1": X_train, "input_2": v_train}, {"output_1": y_1_train, "output_2": y_2_train}, batch_size=64, epochs = epochs,  validation_data=({"input_1": X_valid, "input_2": v_valid}, {"output_1": y_1_valid, "output_2": y_2_valid}))
        with open('model_input_2_output_2___.pkl', 'wb') as files: pickle.dump(model, files)
        # make predictions on test sets
        yhat = model.predict({"input_1": X_test, "input_2": v_test})
        yhat_class = yhat[1].round()

        # calculate accuracy
        acc = accuracy_score(y_2_test, yhat_class)
        print('accuracy_score ','> %.3f' % acc)

        # evaluate model on test set
        mae_1 = mean_absolute_error(y_1_test, yhat[0])
        mae_2 = mean_squared_error(y_1_test, yhat[0])
        print('mean_absolute_error', mae_1, 'mean_squared_error', mae_2)
      
if __name__ == '__main__':
    tr = train()
    tr.train_()


