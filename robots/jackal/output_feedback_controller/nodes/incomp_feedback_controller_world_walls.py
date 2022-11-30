#!/usr/bin/env python3

#import the dependencies
from email.message import EmailMessage
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

class Feedback_2D_Input:
    def __init__(self,K_gains,K_added): 
        self.pub = rospy.Publisher('/u_input', TwistStamped, queue_size=1)
        self.linear_vel =  rospy.Subscriber("/linear_vel", Float64, self.linear_vel_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.apriltags_list = list()
        self.K_gains = np.array(K_gains)
        self.K_added = np.array(K_added)
        self.k_index = 0
        self.selected_apriltag = None
        self.selected_aptag_id = None
        self.apriltag_dist = None
        self.prev_vel = None
        self.position_landmark_inworld_matrix = {0:np.array([[0,1,0,1.8907],[-1,0,0,5.27535],[0,0,1,0.5],[0,0,0,1]]),
                                                 1:np.array([[-1,0,0,7.3294],[0,-1,0,2.79793],[0,0,1,0.5],[0,0,0,1]]),
                                                 2:np.array([[-1,0,0,6.69073],[0,-1,0,-3.370437],[0,0,1,0.5],[0,0,0,1]]),
                                                 3:np.array([[0,-1,0,-3.1056],[1,0,0,-8.2912],[0,0,1,0.5],[0,0,0,1]]),
                                                 4:np.array([[1,0,0,-7.6986],[0,1,0,-2.903],[0,0,1,0.5],[0,0,0,1]]),
                                                 5:np.array([[0,1,0,-4.86078],[-1,0,0,4.4972443],[0,0,1,0.5],[0,0,0,1]]),
                                                 6:np.array([[0,-1,0,1.398419],[1,0,0,2.0676999],[0,0,1,0.5],[0,0,0,1]]),
                                                 7:np.array([[1,0,0,3.3571999],[0,1,0,0.91013699],[0,0,1,0.5],[0,0,0,1]]),
                                                 8:np.array([[1,0,0,2.6263],[0,1,0,-2.40406966],[0,0,1,0.5],[0,0,0,1]]),
                                                 9:np.array([[0,1,0,-1.81357],[-1,0,0,-4.57658],[0,0,1,0.5],[0,0,0,1]]),
                                                 10:np.array([[-1,0,0, -3.99669],[0,-1,0,-2.11309],[0,0,1,0.5],[0,0,0,1]]),
                                                 11:np.array([[-1,0,0,-2.8471093],[0,-1,0,0.9576727],[0,0,1,0.5],[0,0,0,1]]),
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
                    selected_apriltag = at.pose.pose

            #change frame from camera to baselink
            source_frame = "front_realsense_gazebo"
            transform = self.tfBuffer.lookup_transform(source_frame, "base_link",rospy.Time(0), rospy.Duration(1.0))
            # pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
            print('transform',tu.msg_to_se3(transform)) 
            ''' convert the stamped message '''
            self.selected_apriltag = np.dot(tu.msg_to_se3(transform),tu.msg_to_se3(at.pose.pose.pose))
            self.selected_aptag_id = selected_aptag_id 
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.selected_apriltag = None
            self.selected_aptag_id = None

    ## to update the k_index 
    def linear_vel_callback(self,msg):
        if msg:
            print('msg_li_vel',msg)
            threshold = 0.001
            if abs(msg.data) < threshold:
                self.k_index += 1
                # 

    ## compute u using only one apriltag then check how you can include multipe apriltags
    def compute_u(self): # n = num of aptags seen K=[2n,?] y = [2n,1]
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
        if self.selected_aptag_id is not None:
            selected_id = self.selected_aptag_id
            selected_aptag = self.selected_apriltag
            # print("selected_id",selected_id)
            # print("k_index", self.k_index)
            K_gains = self.K_gains[self.k_index*2:self.k_index*2+2,:]
            # print("from_get_state", from_get_state)
            ori_ldmark = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3], (selected_aptag[:3,:3]).T)
            
            # # Rotate the relative distance:
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
            # distance between apriltags and selected tag                
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
            # u = np.concatenate((u.flatten(),[0]))
            print('u',u)
        # # publish control
        #     msg = TwistStamped()
        #     msg.header.stamp = rospy.Time.now()
        #     msg.twist.linear.x = u[0]
        #     msg.twist.linear.y = u[1]
        #     self.pub.publish(msg)
        #     self.prev_vel = u
        # elif self.prev_vel is not None:
        #     msg = TwistStamped()
        #     msg.header.stamp = rospy.Time.now()
        #     msg.twist.linear.x = self.prev_vel[0]
        #     msg.twist.linear.y = self.prev_vel[1]
        #     self.pub.publish(msg)
        #     rospy.logwarn("Publishing old control cause there is no apriltag detected")
        

    
def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

home_dir = os.environ["HOME"]

if __name__ == "__main__":
    rospy.init_node("u_control")
    
    shared_path = os.environ["HOME"]+"/catkin_ws/src/output_feedback_controller/csv/"
    
    K_gains_path= shared_path + "K_gains_new.csv"
    K_gains = read_matrix(K_gains_path)

    K_added_path= shared_path + "K_added_new.csv"
    K_added = read_matrix(K_added_path)

    jackal = Feedback_2D_Input(K_gains,K_added)
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        jackal.compute_u()

    r.sleep()   




 

