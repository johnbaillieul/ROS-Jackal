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
    def __init__(self,K_gains): 
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # self.linear_vel =  rospy.Subscriber("/linear_vel", Float64, self.linear_vel_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.apriltags_list = list()
        self.K_gains = np.array(K_gains)
        # self.K_added = np.array(K_added)
        self.k_index = 0
        # self.u_input = None]
        self.selected_apriltag = None
        self.selected_aptag_id = None
        self.old_vel = None
        self.old_id = 1 # hard_coded
        # self.used_apriltags = [0,1,2,3,4,5,6,7,8,9,10,11] # add the apriltag ids that you used
        self.apriltag_dist = None
        self.k_index_list = {0:5,
                            1:0,
                            2:1,
                            3:2,
                            4:3,
                            5:4,
                            6:5,
                            7:1,
                            8:1,
                            9:3,
                            10:4,
                            11:5,
                            }
 

        self.position_landmark_inworld_matrix = {0:np.array([[1,0,0,1.8907],[0,0,-1,5.27535],[0,1,0,0.5],[0,0,0,1]]),
                                                 1:np.array([[0,0,-1,7.3294],[-1,0,0,2.79793],[0,1,0,0.5],[0,0,0,1]]),
                                                 2:np.array([[0,0,-1,6.69073],[-1,0,0,-3.370437],[0,1,0,0.5],[0,0,0,1]]),
                                                 3:np.array([[-1,0,0,-3.1056],[0,0,1,-8.2912],[0,1,0,0.5],[0,0,0,1]]),
                                                 4:np.array([[0,0,1,-7.6986],[1,0,0,-2.903],[0,1,0,0.5],[0,0,0,1]]),
                                                 5:np.array([[1,0,0,-4.86078],[0,0,-1,4.4972443],[0,1,0,0.5],[0,0,0,1]]),
                                                 6:np.array([[-1,0,0,1.398419],[0,0,1,2.0676999],[0,1,0,0.5],[0,0,0,1]]),
                                                 7:np.array([[0,0,1,3.3571999],[0,1,0,0.91013699],[0,1,0,0.5],[0,0,0,1]]),
                                                 8:np.array([[0,0,1,2.6263],[1,0,0,-2.40406966],[0,1,0,0.5],[0,0,0,1]]),
                                                 9:np.array([[1,0,0,-1.81357],[0,0,-1,-4.57658],[0,1,0,0.5],[0,0,0,1]]),
                                                 10:np.array([[0,0,-1, -3.99669],[-1,0,0,-2.11309],[0,1,0,0.5],[0,0,0,1]]),
                                                 11:np.array([[0,0,-1,-2.8471093],[-1,0,0,0.9576727],[0,1,0,0.5],[0,0,0,1]]),
                                                }
 
    def apriltag_callback(self,msg):
        if msg.detections:
            # '''If there's an AprilTag in the image it selectes the one closest to the apriltag '''
            min_distance = np.inf
            for at in msg.detections:
                dist = np.linalg.norm([at.pose.pose.pose.position.x, at.pose.pose.pose.position.y, at.pose.pose.pose.position.z])
                if dist < min_distance:
                    min_distance = dist
                    selected_aptag_id = at.id[0]

            #change frame from camera to baselink
            source_frame = "front_realsense_gazebo"
            transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
            # print('transform',tu.msg_to_se3(transform))
           
            self.selected_apriltag = np.dot(tu.msg_to_se3(transform),tu.msg_to_se3(at.pose.pose.pose))

            # print('id',selected_aptag_id)
            # print('pose',tu.msg_to_se3(at.pose.pose.pose))
            # print('apriltag',self.selected_apriltag)
            self.selected_aptag_id = selected_aptag_id 
            # print('ap_id',self.selected_aptag_id)
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.selected_apriltag = None
            self.selected_aptag_id = None

    ## compute u using only one apriltag then check how you can include multipe apriltags
    def compute_u(self): # n = num of aptags seen K=[2n,?] y = [2n,1]
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
        if self.selected_apriltag is not None:
            selected_id = self.selected_aptag_id
            
            selected_aptag = self.selected_apriltag
            # print("u_ap",selected_aptag)

            K_gains = self.K_gains[self.k_index*2:self.k_index*2+2,:]
            ## print("from_get_state", from_get_state)
            ori_ldmark = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3], (selected_aptag[:3,:3]).T)
            ## Rotate the relative distance:
            aptag_dist_vector = np.dot(ori_ldmark, selected_aptag[:3,3])[:2].reshape((2,1)); 
            self.apriltag_dist = aptag_dist_vector
            u_1 = None
            for i in range(int(np.size(K_gains,1)/2)):
                if u_1 is None:
                    u_1 = K_gains[:,i*2:i*2+2].dot(aptag_dist_vector)
                else:
                    u_1  = np.hstack((u_1,K_gains[:,i*2:i*2+2].dot(aptag_dist_vector)))
            u_1 = u_1.sum(axis = 1)

            ### 2nd term in u
            ## distance between apriltags and selected tag                
            lj_li = None 
            for value in self.position_landmark_inworld_matrix.values():
                x = value[0,3] - self.position_landmark_inworld_matrix[selected_id][0,3]
                y = value[1,3] - self.position_landmark_inworld_matrix[selected_id][1,3]
                
                if lj_li is None:
                    lj_li = np.vstack((x,y))
                else:
                    lj_li = np.vstack((lj_li,x,y))
        
            u_2 = np.dot(K_gains,lj_li).sum(axis=1)
            u = u_1 + u_2
            return u/np.linalg.norm(K_gains)*np.linalg.norm(u)

    def to_tf(self,pos,ori):
        return np.block([[np.array(ori),pos.reshape((-1,1))],[0,0,0,1]])

# keep track of the unnormalized velocity when checking if it dropped less than a certain threshold
    def compute_input_parse(self):
        aptag_transf = self.selected_apriltag
        selected_id = self.selected_aptag_id

        if selected_id is None:
            #spin in place 
            self.vel.angular.z = -0.5
            self.pub_vel.publish(self.vel)
            return
            # if self.old_vel is not None:
            #     self.pub_vel.publish(self.old_vel) 
            #     print("Using old velocity...")
            #     return 
            # else:
            #     print("Didn't update velocities...")
            #     return
        threshold = 0.001
        u_input = self.compute_u()
        self.old_u = copy.deepcopy(u_input)
        # print('ulen',len(u_input))

        ori = self.robot_pose(aptag_transf,selected_id)
        alpha = 10 # linear coef
        beta = 0.9 # angular coef
        
        linear_velocity = alpha * np.dot(u_input, ori[:2])
        print('Linear_Vel',linear_velocity)
        linear_velocity /= np.linalg.norm(u_input)
        print("linevel",linear_velocity)
        # print("dot",np.dot(u_input, ori[:2])
        self.k_index = self.k_index_list[selected_id]
        print("K_index",self.k_index)
        # if abs(linear_velocity) < threshold:
        #     if self.self.k_index != 5:
        #         self.k_index += 1
        #         print("K_index",self.k_index)
        #     else:
        #         # rospy.loginfo("Finished experiment!") 
        #         return 

        # if selected_id != self.old_id:
        #     if self.k_index != 5:
        #         self.k_index += 1
        #         print("K_index",self.k_index)
        #     else:
        #         # rospy.loginfo("Finished experiment!") 
        #         return 

        if linear_velocity < 0: 
            linear_velocity = 0.5
            angular_velocity = beta*np.cross(ori,u_input/np.linalg.norm(u_input))[2]
        else:
            angular_velocity = beta*np.cross(ori,u_input/np.linalg.norm(u_input))[2]
            
        print('u',u_input)
        print("u_id",selected_id)
        print( 'ori',ori[:2])
        
        self.vel.linear.x = linear_velocity/np.linalg.norm(u_input)
        # print('norm_Linear_Vel',linear_velocity/np.linalg.norm(u_input))
        self.vel.angular.z = angular_velocity 
        print('angular_velocity',angular_velocity )
        # print('linear_vel',  linear_velocity/np.linalg.norm(u_input), "angl vel",angular_velocity)
        self.old_vel = self.vel
        self.pub_vel.publish(self.vel) 
        # self.old_id = selected_id
        
        
## Get the orientation from different apriltags this only gets from the closest one
    def robot_pose(self,aptag_transf,selected_id):                           
        ori = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3],(aptag_transf[:3,:3]).T)
        # print('self.position_landmark_inworld_matrix',self.position_landmark_inworld_matrix[selected_id][:3,:3])
        # print('aptag_transf',aptag_transf[:3,:3])
        
        ori= ori[:3,0].flatten()
        ori[2] = 0
        # print("orien",ori)
        ori/= np.linalg.norm(ori)
        return ori
    
def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

home_dir = os.environ["HOME"]

if __name__ == "__main__":
    rospy.init_node("u_control")
    shared_path = os.environ["HOME"]+"/catkin_ws/src/output_feedback_controller/csv/"
    K_gains_path= shared_path + "K_gains_poly.csv"
    K_gains = read_matrix(K_gains_path)

    # K_added_path= shared_path + "K_added.csv"
    # K_added = read_matrix(K_added_path)

    jackal = Feedback_2D_Input(K_gains)
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        jackal.compute_input_parse()
        # jackal.robot_pose(jackal.selected_apriltag, \
        # jackal.selected_aptag_id)
        # jackal.compute_u()
        # jackal.apriltag_callback()
        r.sleep()   




 

