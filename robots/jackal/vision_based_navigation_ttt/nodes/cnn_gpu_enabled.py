import cv2
import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
import openpyxl
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

class train_5():
    def __init__(self):
        pass
    def train_(self):
        np.random.seed(0)

        X = []
        y = []

        path_tau = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/tau_values/tau_value"   
        path_folder = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/training_images/"
        path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/"
        folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
        print('folders',folders)
        for folder in folders:

            path_images = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/training_images/" + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]
            image_size = 150
            for idx in range(len(images_in_folder)-1) :#
                # print(images_in_folder[idx])
                try:
                    # load the image
                    img_1 = cv2.imread(path_images + images_in_folder[idx],0)
                    img_1 = cv2.resize(img_1,(image_size ,image_size ))
                    image_as_array_1 = np.array(img_1)
                    image_as_array_1 = image_as_array_1.reshape(image_size ,image_size ,1)
                    # print('1', image_as_array_1.shape)

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1],0)
                    img_2 = cv2.resize(img_2,(image_size ,image_size ))
                    image_as_array_2 = np.array(img_2)
                    
                    image_as_array_2 = image_as_array_2.reshape(image_size ,image_size ,1)
                    # print('2',image_as_array_2.shape)
                    # print(self.total_imgs)
                    
                    img = np.stack([img_1, img_2], 2)
                    
                    # print(img)
                    # add our image to the dataset
                    X.append(img)

                    # # get velocity
                    # vel = float(folder.split('_')[4])
                    # # print('ve',vel)
                    # velocity.append([vel])

                    # retrive the direction from the filename
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    y.append(np.asarray(tau_values))

                except Exception as inst:
                    print(idx)
                    print(inst)
         
        X = np.asarray(X)
        print('x',X.shape)
        y = np.asarray(y)
        print('y',y.shape)
        # velocity = np.asarray(velocity)
        
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

        # v_train = velocity[ind[0:train_index]]
        # v_valid = velocity[ind[train_index:train_index+valid_index]]
        # v_test = velocity[ind[train_index+valid_index:]]

        # tf.debugging.set_log_device_placement(True)


        # # Convolutional Neural Network
        input_1 = keras.layers.Input(shape=(image_size ,image_size ,2))
        conv1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(input_1)
        pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(pool1)
        pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(pool2)
        pool3 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(pool3)

        # input_2 = keras.layers.Input(shape=(1))

        # merge input models
        # merge = keras.layers.concatenate([flat, input_2])  

        hidden1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(flat)
        drop_1 = keras.layers.Dropout(0.25)
        hidden2 = keras.layers.Dense(225, activation='relu',kernel_initializer='he_uniform')(hidden1)
        drop_1 = keras.layers.Dropout(0.25)(hidden2)
        output = keras.layers.Dense(5,kernel_initializer='he_uniform')(drop_1)

        model = keras.models.Model(inputs=[input_1], outputs=output)
        model.compile(optimizer = 'adam', loss = 'mae', metrics = 'accuracy')

        # summarize layers
        print(model.summary())
        # # plot graph
        model_name = 'model_5_without_v_flag_img_size_' + str(image_size) 
        keras.utils.plot_model(model, model_name + '.png', show_shapes=True)
      
        ## train the model
        model.fit({"input_1": X_train}, y_train, batch_size=64, epochs = 100,  validation_data=({"input_1": X_valid}, y_valid))
        with open(path + 'vision_based_navigation_ttt/trained_model_parameters/' + model_name + '.pkl', 'wb') as files: pickle.dump(model, files)

        # print(model.predict(X_test)[8])
        test_loss, test_acc = model.evaluate({"input_1": X_test}, y_test)
        print('test loss: {}, test accuracy: {}'.format(test_loss, test_acc) )

class train_8():
    def __init__(self):
        pass
    def train_(self):
        np.random.seed(0)
        img_size = 300
        X = []
        y_1 = []
        y_2 = []
        velocity = []

        path_tau = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/tau_values/tau_value"   
        path_folder = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/training_images/"
        path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/"

        folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
        # print('ggggggggggggg',folders)
        for folder in folders:

            # print('fol',folder)
            path_images = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/training_images/" + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]

            for idx in range(len(images_in_folder)-1) : #len(images_in_folder)-1
                # print(images_in_folder[idx])
                try:
                    # load the image
                    img_1 = cv2.imread(path_images + images_in_folder[idx],0)
                    img_1 = cv2.resize(img_1,(img_size,img_size))
                    # print('1', image_as_array_1.shape)

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1],0)
                    img_2 = cv2.resize(img_2,(img_size,img_size))
                    
                    img = np.stack([img_1, img_2], 2)
                    
                    # print(img)
                    # add our image to the dataset
                    X.append(img)

                    # # get velocity
                    # vel = float(folder.split('_')[4])
                    # # print('ve',vel)
                    # velocity.append([vel])

                    # retrive the direction from the filename
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx+1].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    y_1.append(np.asarray(tau_values))
                    # print(y_1)

                    y_temp = []
                    for i in tau_values:
                        if i == -1:
                            y_temp.append(0)
                        else:
                            y_temp.append(1)
                    y_2.append(y_temp)

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
        # velocity = np.asarray(velocity)
        
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

        # v_train = velocity[ind[0:train_index]]
        # v_valid = velocity[ind[train_index:train_index+valid_index]]
        # v_test = velocity[ind[train_index+valid_index:]]

        # # Convolutional Neural Network
        input_1 = keras.layers.Input(shape=(img_size,img_size,2))
        conv1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(input_1)
        pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(pool1)
        pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(pool2)
        pool3 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(pool3)
        hidden_1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(flat)
        output_2 = keras.layers.Dense(5, activation= 'sigmoid', name = 'output_2')(hidden_1)
        # input_2 = keras.layers.Input(shape=(1))

        # merge input models
        # merge = keras.layers.concatenate([flat, input_2])  

        
        hidden2 = keras.layers.Dense(225, activation='relu',kernel_initializer='he_uniform')(hidden_1)
        drop_2 = keras.layers.Dropout(0.25)(hidden2)
        output_1 = keras.layers.Dense(5, name = 'output_1')(drop_2)

        model = keras.models.Model(inputs=[input_1], outputs=[output_1, output_2])

        # summarize layers
        print(model.summary())
        # # plot graph
        model_name = 'model_8_without_v_flag_img_size_' + str(img_size) 
        keras.utils.plot_model(model, model_name + '.png', show_shapes=True)
        
        model.compile(optimizer = 'adam', loss = {"output_1" :'mae', "output_2" :'binary_crossentropy'})
        # , metrics = [['mean_absolute_error']]
        epochs = 100
        ## train the model
        model.fit({"input_1": X_train}, {"output_1": y_1_train, "output_2": y_2_train}, batch_size=64, epochs = epochs,  validation_data=({"input_1": X_valid}, {"output_1": y_1_valid, "output_2": y_2_valid}))
        
        with open(path + 'vision_based_navigation_ttt/trained_model_parameters/' + model_name + '.pkl', 'wb') as files: pickle.dump(model, files)

        # make predictions on test sets
        yhat = model.predict({"input_1": X_test})
        # print(yhat)
        # print(yhat[0], yhat[1])
        # round pred
        yhat_class = yhat[1].round()
        # calculate accuracy
        acc = accuracy_score(y_2_test, yhat_class)
        # store result 
        print('accuracy_score ','> %.3f' % acc)

        # evaluate model on test set
        mae_1 = mean_absolute_error(y_1_test, yhat[0])
        mae_2 = mean_squared_error(y_1_test, yhat[0])
        print('mean_absolute_error', mae_1, 'mean_squared_error', mae_2)

if __name__ == '__main__':
    tr = train_8()
    tr.train_()
   

