#!/usr/bin/env python

import numpy as np
import os

def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

def write_matrix(csv_dir,mat):
    with open(csv_dir,'w') as f:
        np.savetxt(f,mat,delimiter=',')

K1 = np.array([[0.0942,0,0.4207,0,0.4850,0],\
                [0,0.5822,0,0.2089,0,0.2089]])

K2 = np.array([[0.9997,0,0,0,0,0],\
                [0,1.0138,0,-0.0225,0,0.0087]])


if __name__ == "__main__":
    
    # Remember: transform matrix does local->global

    root_dir = os.environ["HOME"]+"/catkin_ws/src/creates_iros/csv/"
    transform_dir = root_dir + "transform_matrix.csv"
    transform_matrix = read_matrix(transform_dir)

    print(transform_matrix)

    initial_landmarks_dir = root_dir + '/landmark_positions.csv'
    goal_landmarks_dir = root_dir + '/local_landmarks.csv'
    
    initial_landmarks = read_matrix(initial_landmarks_dir).T
    if initial_landmarks.shape[0] == 3:
        initial_landmarks = np.concatenate((initial_landmarks,np.ones((1,initial_landmarks.shape[1]))),axis=0)
    
    goal_landmarks = np.linalg.inv(transform_matrix).dot(initial_landmarks).T[:,:2]
    write_matrix(goal_landmarks_dir,goal_landmarks)

    print(goal_landmarks)
