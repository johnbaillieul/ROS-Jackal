#!/usr/bin/env python

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
        self.used_apriltags = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] # add the apriltag ids that you used
        self.position_landmark_inworld_matrix = {}
        self.apriltag_dist = None
        
        # qhich cell do each landmark belong to 
        self.K_to_cell = {0:'L1',1:'L6',2:'L6',3:'L3',4:'L1',5:'L1',6:'L2',7:'L2',8:'L3',9:'L3',10:'L3',11:'L4',12:'L5',13:'L5',14:'L4',15:'L2',16:'L4',17:'L6',18:'L5',19:'L6'}
        self.cell_to_landmark = {'L1':[0,4,5,6,7],'L2':[6,7,8,9,15],'L3':[8,9,10,11,3],'L4':[16,11,14,12],'L5':[12,1,18,13],'L6':[1,2,17,19,4,5]}
        self.current_cell = 0

        # self.cell_to_landmark = {'L1':[0,13,6,7,16], 'L2':[1,5,4,14],'L3':[11,2,10,15],'L4':[3,12,8,9]}
        # self.K_to_cell = {2:'L1',3:'L2',4:'L2',5:'L2',6:'L2',7:'L3',8:'L3',9:'L3',10:'L3',11:'L4',12:'L4',13:'L1',14:'L4'}


    def get_rot_matrix_aptags(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        model = GetModelStateRequest()
        for id in self.used_apriltags:
            model.model_name = 'apriltag'+str(id)
            result = get_model_srv(model)
            # print('id',id,result.pose.position)
            self.position_landmark_inworld_matrix[id] = tu.msg_to_se3(result.pose)
        
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
            pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
            ''' convert the stamped message '''
            self.selected_apriltag = tu.msg_to_se3(pose_transformed.pose)
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
            if msg < threshold:
                self.k_index += 1
                self.current_cell += 1

    ## compute u using only one apriltag then check how you can include multipe apriltags
    def compute_u(self): # n = num of aptags seen K=[2n,?] y = [2n,1]
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
        if self.selected_aptag_id is not None and self.selected_aptag_id in self.cell_to_landmark['L'+str(self.current_cell + 1)]:
            selected_id = self.selected_aptag_id
            selected_aptag = self.selected_apriltag
            # print("selected_id",selected_id)
            # print("k_index", self.k_index)
            K_gains = self.K_gains[self.k_index*2:self.k_index*2+2,:]
            K_gains = K_gains[~np.isnan(K_gains)].reshape((2,-1))
            # print('kgains',K_gains)
            # print("from_get_state", from_get_state)
            ori_ldmark = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3], np.linalg.inv(selected_aptag[:3,:3]))
            
            # # Rotate the relative distance:
            aptag_dist_vector = np.dot(ori_ldmark, selected_aptag[:3,3])[:2].reshape((2,1)); 
            self.apriltag_dist = aptag_dist_vector
            u_1 = None
            for i in range(np.size(K_gains,1)/2):
                if u_1 is None:
                    u_1 = K_gains[:,i*2:i*2+2].dot(aptag_dist_vector)
                    
                else:
                    u_1  = np.hstack((u_1,K_gains[:,i*2:i*2+2].dot(aptag_dist_vector)))
            u_1 = u_1.sum(axis = 1)
     
            ### 2nd term in u
            # distance between apriltags and selected tag                
            lj_li = None 
            # print( 'celllndmak', self.cell_to_landmark['L'+str(self.current_cell + 1)])
            for lndmark in self.cell_to_landmark['L'+str(self.current_cell + 1)]:
                x = self.position_landmark_inworld_matrix[lndmark][0,3] - self.position_landmark_inworld_matrix[selected_id][0,3]
                y = self.position_landmark_inworld_matrix[lndmark][1,3] - self.position_landmark_inworld_matrix[selected_id][1,3]
                
                if lj_li is None:
                    lj_li = np.vstack((x,y))
                else:
                    lj_li = np.vstack((lj_li,x,y))
            # print('lj',lj_li)
            u_2 = np.dot(K_gains,lj_li).sum(axis=1)
            u = u_1 + u_2
            # u = np.concatenate((u.flatten(),[0]))
            print('u_ola',u)
        # publish control
            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.twist.linear.x = u[0]
            msg.twist.linear.y = u[1]
            self.pub.publish(msg)

    
def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

home_dir = os.environ["HOME"]

if __name__ == "__main__":
    rospy.init_node("u_control")
    
    shared_path = os.environ["HOME"]+"/catkin_ws/src/output_feedback_controller/csv/"
    
    K_gains_path= shared_path + "K_gains_dif.csv"
    K_gains = read_matrix(K_gains_path)

    K_added_path= shared_path + "K_added_dif.csv"
    K_added = read_matrix(K_added_path)

    jackal = Feedback_2D_Input(K_gains,K_added)
    jackal.get_rot_matrix_aptags()
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        jackal.compute_u()

    r.sleep()   




 

