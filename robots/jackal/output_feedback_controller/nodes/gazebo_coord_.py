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
import tf.transformations as tr

class gazebo_callback:
    def __init__(self): 
        self.position_landmark_inworld_matrix = {}
        self.used_apriltags = [1]
        self.selected_apriltag = None
        self.selected_aptag_id = None
        # gets the location of the apriltags in gazebo using model state service
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def get_rot_matrix_aptags(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        model = GetModelStateRequest()
        for id in self.used_apriltags:
            model.model_name = 'apriltag'+str(id)
            trans = get_model_srv(model)
            # print('id',id,result.pose.position)
            
            T = tr.translation_matrix([trans.pose.position.x, trans.pose.position.y, trans.pose.position.z])
            R = tr.quaternion_matrix([trans.pose.orientation.x, trans.pose.orientation.y, trans.pose.orientation.z, trans.pose.orientation.w])
            self.position_landmark_inworld_matrix[id] = np.dot(T, R)
        # print('posiiton', self.position_landmark_inworld_matrix)


    def apriltag_callback(self,msg):
        if msg.detections:
            # '''If there's an AprilTag in the image it selectes the one closest to the apriltag '''
            min_distance = np.inf
            for at in msg.detections:
                dist = np.linalg.norm([at.pose.pose.pose.position.x, at.pose.pose.pose.position.y, at.pose.pose.pose.position.z])
                if dist < min_distance:
                    min_distance = dist
                    selected_aptag_id = at.id[0]
                    selected_apriltag = at.pose.pose
                    
            print('selected_aptag_id', selected_aptag_id)

            # # Transformation of camera in baselink
            source_frame = "front_realsense_gazebo"
            frame = "tag_1"
            
            trans = self.tfBuffer.lookup_transform(source_frame, frame, rospy.Time(0), rospy.Duration(1.0))
            T = tr.translation_matrix([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            R = tr.quaternion_matrix([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            
            # # Convert the transform to a homogeneous transformation matrix
            transform_aptag_in_cam = np.dot(T, R)
            print("transform_aptag_in_cam", transform_aptag_in_cam)
            
            # aprtag_gazebo in world 
            transform_aptaggz_in_world =  self.position_landmark_inworld_matrix[1]
            print('transform_aptaggazebo_in_world_nothing', transform_aptaggz_in_world)

            # aptag in aptaggazebo
            transform_aptag_in_aptaggz =  np.block([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])
            print('transform_aptag_in_aptaggz', transform_aptag_in_aptaggz)
     
            # Define the 4x4 homogeneous matrix
            M = np.dot(np.dot(transform_aptaggz_in_world, transform_aptag_in_aptaggz), np.linalg.inv(transform_aptag_in_cam)) 
            print('M', M)

            [M[3,0],M[3,1]].reshape(2,1)
            self.selected_aptag_id = selected_aptag_id 
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.selected_apriltag = None
            self.selected_aptag_id = None


home_dir = "/home"

if __name__ == "__main__":
    rospy.init_node("gaz")
    shared_path = home_dir + "/catkin_ws/src/output_feedback_controller/csv/"

    jackal = gazebo_callback()
    jackal.get_rot_matrix_aptags()
    print(jackal.position_landmark_inworld_matrix)
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        r.sleep()  



