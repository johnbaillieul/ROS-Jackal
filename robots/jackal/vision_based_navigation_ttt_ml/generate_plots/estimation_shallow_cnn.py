import numpy as np
import os
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import autokeras as ak

#Dataset used:
path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/"
 
path_image = path + "test_results_img/training_images_327_v_1/"

first_image_no = 48803
last_image_no = 49071

range_frames = (first_image_no, last_image_no)

model = tf.keras.models.load_model(path + "trained_model_parameters/custom_ml_updated_data.h5", compile=False, custom_objects=ak.CUSTOM_OBJECTS)
model.compile(optimizer = 'adam', loss = 'mae', metrics = ['MeanSquaredError', 'mean_absolute_error']) #Paste it here

# model.summary()

#Load images
res = []
vel = 1
vel = np.asarray([vel])

for frame_no in range(range_frames[0], range_frames[1]):
    
    print(frame_no)
    path_img1 = path_image + str(frame_no) + ".png" 
    path_img2 = path_image + str(frame_no + 1) + ".png"
    img_size = 250
    
    img_1 = cv2.imread(path_img1)
    img_1 = cv2.resize(img_1,(img_size,img_size))
    
    img_2 = cv2.imread(path_img2)
    print("Image2", np.shape(img_2))
    img_2 = cv2.resize(img_2,(img_size,img_size))
    
    img = np.concatenate([img_1, img_2], 2)
    img = tf.expand_dims(img, axis=0)

    # print(img)
    pred = model.predict({"input_1": img, "input_2": vel})[0]
    print(pred)
    res.append(pred)
    

tau_left_pred = [tmp[0] for tmp in res]
tau_left_pred = [tmp[1] for tmp in res]
tau_center_pred = [tmp[2] for tmp in res]
tau_right_pred = [tmp[3] for tmp in res]
tau_rright_pred = [tmp[4] for tmp in res]

np.save(path +'shallow_results/shallow_tau_le_HG.npy', tau_left_pred)
np.save(path +'shallow_results/shallow_tau_l_HG.npy', tau_left_pred)
np.save(path +'shallow_results/shallow_tau_c_HG.npy', tau_center_pred)
np.save(path +'shallow_results/shallow_tau_r_HG.npy', tau_right_pred)
np.save(path +'shallow_results/shallow_tau_re_HG.npy', tau_rright_pred)