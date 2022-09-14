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

class Feedback_2D_Input:
    def __init__(self,K_gains,K_added): 
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # self.linear_vel =  rospy.Subscriber("/linear_vel", Float64, self.linear_vel_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.apriltags_list = list()
        self.K_gains = np.array(K_gains)
        self.K_added = np.array(K_added)
        self.k_index = 0
        # self.u_input = None
        self.selected_apriltag = None
        self.selected_aptag_id = None
        # self.used_apriltags = [0,1,2,3,4,5,6,7,8,9,10,11] # add the apriltag ids that you used
        self.apriltag_dist = None
        self.position_landmark_inworld_matrix = {1:np.array([[1,0,0,-8.3137],[0,1,0,-5.89405],[0,0,1,0.5],[0,0,0,1]]),
                                                 2:np.array([[1,0,0,-8.25174],[0,1,0,-1.70236],[0,0,1,0.5],[0,0,0,1]]),
                                                 3:np.array([[1,0,0,-8.20742],[0,1,0,1.44827],[0,0,1,0.5],[0,0,0,1]]),
                                                 4:np.array([[0,1,0,-5.59215],[-1,0,0,3.13622],[0,0,1,0.5],[0,0,0,1]]),
                                                 5:np.array([[0,1,0,-2.523],[-1,0,0,3.28701],[0,0,1,0.5],[0,0,0,1]]),
                                                 6:np.array([[-1,0,0,1.29401],[0,-1,0,1.90886],[0,0,1,0.5],[0,0,0,1]]),
                                                 7:np.array([[-1,0,0,1.28591],[0,-1,0,-2.18859],[0,0,1,0.5],[0,0,0,1]]),
                                                 8:np.array([[-1,0,0,1.24712],[0,-1,0,-6.21105],[0,0,1,0.5],[0,0,0,1]]),
                                                 9:np.array([[0,-1,0,-2.48163],[1,0,0,-8.77517],[0,0,1,0.5],[0,0,0,1]]), 
                                                 10:np.array([[0,-1,0,-5.67756],[1,0,0,-8.7319],[0,0,1,0.5],[0,0,0,1]]),
                                                 11:np.array([[-1,0,0,-4.03817],[0,-1,0,-5.24104],[0,0,1,0.5],[0,0,0,1]]),
                                                 12:np.array([[-1,0,0,-4.16755],[0,-1,0,-2.12336],[0,0,1,0.5],[0,0,0,1]]),    
                                                 13:np.array([[1,0,0,-2.11855],[0,1,0,-2.11572],[0,0,1,0.5],[0,0,0,1]]),
                                                 14:np.array([[1,0,0,-2.06867],[0,1,0,-4.1917],[0,0,1,0.5],[0,0,0,1]]),
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
            transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
            # pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
            ''' convert the stamped message '''
            
            self.selected_apriltag = np.dot(tu.msg_to_se3(transform),tu.msg_to_se3(at.pose.pose.pose))
            self.selected_aptag_id = selected_aptag_id 
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.selected_apriltag = None
            self.selected_aptag_id = None

    ## compute u using only one apriltag then check how you can include multipe apriltags
    def compute_u(self): # n = num of aptags seen K=[2n,?] y = [2n,1]
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
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
        return u
        
    def to_tf(self,pos,ori):
        return np.block([[np.array(ori),pos.reshape((-1,1))],[0,0,0,1]])

# keep track of the unnormalized velocity when checking if it dropped less than a certain threshold
    def compute_input_parse(self):
        aptag_transf = self.selected_apriltag
        selected_id = self.selected_aptag_id

        if selected_id is None:
            print("Didn't update velocities...")
            return

        threshold = 0.001
        u_input = self.compute_u()
  
        # print('ulen',len(u_input))
        ori = self.robot_pose(aptag_transf,selected_id)
        alpha = 0.5 # linear coef
        beta = 0.2 # angular coef
        print('u',u_input)
        # print( 'ori',ori[:2])
        linear_velocity = alpha * np.dot(u_input, ori[:2])
        if abs(linear_velocity) < threshold:
                self.k_index += 1
        print('Linear_Vel',linear_velocity)
        if linear_velocity < 0: 
            linear_velocity = 0.5
            angular_velocity = beta*np.cross(ori,u_input/np.linalg.norm(u_input))[2]

        else:
            angular_velocity = beta*np.cross(ori,u_input/np.linalg.norm(u_input))[2]

        self.vel.linear.x = linear_velocity/np.linalg.norm(u_input)
        self.vel.angular.z = angular_velocity 
        self.pub_vel.publish(self.vel) 
        print('linear',self.vel.linear.x)
        print('angular',angular_velocity)
        print('reached')
            
## Get the orientation from different apriltags this only gets from the closest one
    def robot_pose(self,aptag_transf,selected_id):                           
        ori = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3],\
            (aptag_transf[:3,:3]).T)
        # print('s_id',selected_id)
        # print('tf', self.position_landmark_inworld_matrix[selected_id])
        # print('state', ori_ldmark)
        ori= ori[:3,0].flatten()
        ori[2] = 0
        ori/= np.linalg.norm(ori)
        return ori
    
def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

home_dir = os.environ["HOME"]

if __name__ == "__main__":
    rospy.init_node("u_control")
    
    shared_path = os.environ["HOME"]+"/catkin_ws/src/output_feedback_controller/csv/"
    
    K_gains_path= shared_path + "K_gains.csv"
    K_gains = read_matrix(K_gains_path)

    K_added_path= shared_path + "K_added.csv"
    K_added = read_matrix(K_added_path)

    jackal = Feedback_2D_Input(K_gains,K_added)
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        jackal.compute_input_parse()
    r.sleep()   




 

