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
import tf.transformations as tr

class Feedback_2D_Input:
    def __init__(self,K_gains): 
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.apriltags_list = list()
        self.K_gains = np.array(K_gains)
        # self.K_added = np.array(K_added)
        self.k_index = 0
        self.position_landmark_inworld_matrix = {}
        self.selected_apriltag = None
        self.selected_aptag_id = None
        self.used_apriltags = [0,1,2,3,4,5,6,7,8,9,10,11] # add the apriltag ids that you used
        self.apriltag_dist = None

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
        # print('position', self.position_landmark_inworld_matrix)

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

            source_frame = "front_realsense_gazebo"
            frame = "tag_" + str(selected_aptag_id)
            
            trans = self.tfBuffer.lookup_transform(source_frame, frame, rospy.Time(0), rospy.Duration(1.0))
            T = tr.translation_matrix([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            R = tr.quaternion_matrix([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            
            # # Convert the transform to a homogeneous transformation matrix
            transform_aptag_in_cam = np.dot(T, R)
            # print("transform_aptag_in_cam", transform_aptag_in_cam)
            
            self.selected_apriltag = transform_aptag_in_cam
            self.selected_aptag_id = selected_aptag_id 
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.selected_apriltag = None
            self.selected_aptag_id = None

    ## compute u using only one apriltag then check how you can include multipe apriltags
    def compute_u(self): 
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
        selected_id = self.selected_aptag_id
        selected_aptag = self.selected_apriltag
    
        K_gains = self.K_gains[self.k_index*2:self.k_index*2+2,:]

        # aprtag_gazebo in world 
        transform_aptaggz_in_world =  self.position_landmark_inworld_matrix[selected_id]
        # print('transform_aptaggazebo_in_world_nothing', transform_aptaggz_in_world)

        # aptag in aptaggazebo
        transform_aptag_in_aptaggz =  np.block([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]])
        # print('transform_aptag_in_aptaggz', transform_aptag_in_aptaggz)
    
        # Define the 4x4 homogeneous matrix
        M = np.dot(np.dot(transform_aptaggz_in_world, transform_aptag_in_aptaggz), np.linalg.inv(selected_aptag)) 
        # print('M', M)

        orientation = M[:3,0].flatten()
        print('orientation ggggg',orientation)
        orientation[2] = 0
        orientation /= np.linalg.norm(orientation)
        
        aptag_dist_vector = np.array([M[0,3], M[1,3]]).reshape(2,1)
        # print('aptag_dist_vector',aptag_dist_vector)
        
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
        # print('u',u)
        return u, orientation
        
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
        u_input, ori = self.compute_u()
  
        # print('ulen',len(u_input))
        # ori = self.robot_pose(aptag_transf,selected_id)
        alpha = 0.05 # linear coef
        beta = 0.2 # angular coef
        print('u @@@@@@@@@@@',u_input)
        print( 'ori',ori[:2])
        linear_velocity = alpha * np.dot(u_input, ori[:2])
        if abs(linear_velocity) < threshold:
                self.k_index += 1
        print('Linear_Vel @@@@@@@@@',linear_velocity)
        if linear_velocity < 0: 
            linear_velocity = 0.5
            angular_velocity = beta*np.cross(ori,u_input/np.linalg.norm(u_input))[2]

        else:
            angular_velocity = beta*np.cross(ori,u_input/np.linalg.norm(u_input))[2]

        self.vel.linear.x = linear_velocity/np.linalg.norm(u_input)
        self.vel.angular.z = angular_velocity 
        self.pub_vel.publish(self.vel) 
        # print('linear',self.vel.linear.x)
        print('angular',angular_velocity)
        # print('reached')
            
## Get the orientation from different apriltags this only gets from the closest one
    def robot_pose(self,aptag_transf,selected_id):                           
        ori = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3],\
            (aptag_transf[:3,:3]).T)
        # print('s_id',selected_id)
        # print('tf', self.position_landmark_inworld_matrix[selected_id])
        # print('state', ori_ldmark)
        ori= ori[:3,2].flatten()
        ori[2] = 0
        ori/= np.linalg.norm(ori)
        return ori
    
def read_matrix(csv_dir):
    with open(csv_dir,'r') as f:
        return np.genfromtxt(f,delimiter=',')

home_dir = "/home"

if __name__ == "__main__":
    rospy.init_node("u_control")
    
    shared_path = home_dir + "/catkin_ws/src/output_feedback_controller/csv/"
    
    K_gains_path= shared_path + "K_gains_poly.csv"
    K_gains = read_matrix(K_gains_path)

    # K_added_path= shared_path + "K_added.csv"
    # K_added = read_matrix(K_added_path)

    jackal = Feedback_2D_Input(K_gains)
    jackal.get_rot_matrix_aptags()

    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        jackal.compute_input_parse()
    r.sleep()   




 

