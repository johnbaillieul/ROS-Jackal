#!/usr/bin/env python3
#import the dependencies
import rospy
from geometry_msgs.msg import Twist, Pose, TwistStamped
from apriltag_ros.msg import AprilTagDetectionArray
import transformation_utilities as tu
import numpy as np
import tf2_ros 
import tf2_geometry_msgs
import os
from std_msgs.msg import Float64, Header
import pandas as pd
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import copy
import math

class Feedback_2D_Input:
    def __init__(self): 
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # self.linear_vel =  rospy.Subscriber("/linear_vel", Float64, self.linear_vel_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.apriltags_list = list()
  
        self.k_index = 0
      
        self.selected_apriltag = None
        self.selected_aptag_id = None
        self.old_vel = None
        # self.used_apriltags = [0,1,2,3,4,5,6,7,8,9,10,11] # add the apriltag ids that you used
        self.apriltag_dist = None
        self.position_landmark_inworld_matrix = {13:np.array([[1,0,0,1.3],[0,0,-1,1.32],[0,1,0,0.5],[0,0,0,1]]),
                                                }
    def apriltag_callback(self,msg):
        if msg.detections:
            at = msg.detections[0]
            source_frame = "front_realsense_gazebo"
            transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
            self.selected_apriltag = np.dot(tu.msg_to_se3(transform),tu.msg_to_se3(at.pose.pose.pose))
            self.selected_aptag_id = at.id[0]
            print('ap_id',self.selected_aptag_id)
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.selected_apriltag = None
            self.selected_aptag_id = None

    ## Get the orientation from different apriltags this only gets from the closest one
    def robot_pose(self): 
        if self.selected_apriltag  is not None:
            aptag_transf = self.selected_apriltag 
            selected_id =  self.selected_aptag_id                    
            ori = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3],(aptag_transf[:3,:3]).T)
            print('self.position_landmark_inworld_matrix',self.position_landmark_inworld_matrix[selected_id][:3,:3])
            print('aptag_transf',aptag_transf[:3,:3])
            print("ori",ori)
            ori= ori[:3,0].flatten()
            ori[2] = 0
            print('orib',ori)
            ori/= np.linalg.norm(ori)
            # angular_velocity = np.cross(ori,u_input/np.linalg.norm(u_input))[2]
            print('ori',ori)


if __name__ == "__main__":
    rospy.init_node("ori")
    r = rospy.Rate(10)
    jackal = Feedback_2D_Input()
   
    while not rospy.is_shutdown(): 

        jackal.robot_pose()
        
        r.sleep()   
