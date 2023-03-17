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

class Feedback_2D_Input:
    def __init__(self,K_gains):#,K_added): 
        self.pub = rospy.Publisher('/u_input', TwistStamped, queue_size=1)
        self.linear_vel =  rospy.Subscriber("/linear_vel", Float64, self.linear_vel_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.apriltags_list = list()
        self.K_gains = np.array(K_gains)
        # self.K_added = np.array(K_added)
        self.k_index = 0
        self.used_apriltags = [0,1,2,3,4,5,6,7,8,9,10,11] # add the apriltag ids that you used
        self.position_landmark_inworld_matrix = {}
        self.seen_aptags_id = []
        self.seen_aptags_tf = []

    # gets the location of the apriltags in gazebo using model state service
    def get_rot_matrix_aptags(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        model = GetModelStateRequest()
        for id in self.used_apriltags:
            model.model_name = 'apriltag'+str(id)
            result = get_model_srv(model)
            # print('id',id,result.pose.position)

            self.position_landmark_inworld_matrix[id] = tu.msg_to_se3(result.pose)
        # print('posiiton', self.position_landmark_inworld_matrix)
        
    def apriltag_callback(self,msg):
        if msg.detections:
            # '''If there's an AprilTag in the image it selectes the one closest to the apriltag '''
            for at in msg.detections:
                if at.id[0] not in self.seen_aptags_id:
                    self.seen_aptags_id.append(at.id[0])
                    #change frame from camera to baselink
                    source_frame = "front_realsense_gazebo"
                    transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
                    # print('tf',tu.msg_to_se3(transform))
                    # pose_transformed = tf2_geometry_msgs.do_transform_pose(at.pose.pose, transform)
                    ''' convert the stamped message '''
                    # print('ptrans',tu.msg_to_se3(pose_transformed.pose))
                    # print('olatrial', np.dot(tu.msg_to_se3(transform),tu.msg_to_se3(at.pose.pose.pose)))
                    self.seen_aptags_tf.append(np.dot(tu.msg_to_se3(transform),tu.msg_to_se3(at.pose.pose.pose)))
        else:
            rospy.logwarn("Can't find an AprilTag in the image!")
            self.seen_aptags_tf = None
            self.seen_aptags_id = None 

    ## to update the k_index 
    def linear_vel_callback(self,msg):
        if msg:
            threshold = 0.001
            if msg.data < threshold:
                self.k_index += 1
                # print('self.k_index',self.k_index)
                
                

    ## compute u using only one apriltag then check how you can include multipe apriltags
    def compute_u(self): # n = num of aptags seen K=[2n,?] y = [2n,1]
        ### eq u = sum_j [K_j(l_i - x)] + sum_j [K_j(l_j - li)]
        # take only the rows reprsenting the cell= we are in
        K_gains = copy.deepcopy(self.K_gains)
        K_gains = K_gains[self.k_index*2:self.k_index*2+2,:]
     
        K_gains_mod = copy.deepcopy(self.K_gains)
        K_gains_mod = K_gains_mod[self.k_index*2:self.k_index*2+2,:]

        seen_aptags_id = copy.deepcopy(self.seen_aptags_id)
        # print('seen',len(seen_aptags_id))

        seen_aptags_tf = copy.deepcopy(self.seen_aptags_tf)

        unseen_aptags = [x for x in self.used_apriltags if x not in seen_aptags_id]
        # print('unseen',unseen_aptags)
# 
        last_seen_aptag_id = seen_aptags_id[len(seen_aptags_id)-1]
        u_1 = 0
        u_2 = 0
        u_3 = 0
        
        if len(seen_aptags_id) != 0:
            if len(seen_aptags_id) > 1:
                for i in range(len(seen_aptags_id) - 1):
                    ori_ldmark = np.dot(self.position_landmark_inworld_matrix[seen_aptags_id[i]][:3,:3], \
                    np.linalg.inv(seen_aptags_tf[i][:3,:3]))
                    # # Rotate the relative distance:
                    aptag_dist_vector = np.dot(ori_ldmark, seen_aptags_tf[i][:3,3])[:2].reshape((2,1)); 
                    u_1 += np.dot(K_gains[:,seen_aptags_id[i]*2:seen_aptags_id[i]*2+2],aptag_dist_vector )
                    # print("range",list(range(i*2,i*2+2)) 
                    # print("K_gains_mod:1", np.shape(K_gains_mod))
                    K_gains_mod = np.delete(K_gains_mod,list(range(seen_aptags_id[i]*2,seen_aptags_id[i]*2+2)), axis = 1)
                    # print("K_gains_mod:2", np.shape(K_gains_mod))
                # print('u1',u_1)
            
            # 2nd term
            last_seen_ori_ldmark = np.dot(self.position_landmark_inworld_matrix[last_seen_aptag_id][:3,:3], \
            np.linalg.inv(seen_aptags_tf[np.size(seen_aptags_id)-1][:3,:3]))
            # # Rotate the relative distance:
            last_seen_aptag_dist_vector = np.dot(last_seen_ori_ldmark, seen_aptags_tf[np.size(seen_aptags_id)-1][:3,3])[:2].reshape((2,1));  
            
            for i in range(int(np.shape(K_gains_mod)[1] / 2)- 1):
                # print('i',i)
                # print('K_gains_mod',np.shape(K_gains_mod))
                u_2 += np.dot(K_gains_mod[:,i*2:i*2+2],last_seen_aptag_dist_vector)
            # print('u2',u_2)

            ### 3rd term in u
            # distance between unseen apriltags and last seen tag              
            lj_li = None 
            # print('unseen',np.size(unseen_aptags))
            for id in unseen_aptags:
                # print("id:",id)
                x = self.position_landmark_inworld_matrix[id][0,3] - \
                self.position_landmark_inworld_matrix[last_seen_aptag_id][0,3]
                 # print('x:', x)
                y = self.position_landmark_inworld_matrix[id][1,3] - \
                self.position_landmark_inworld_matrix[last_seen_aptag_id][1,3]
                # print('y:',y)
                
                if lj_li is None:
                    lj_li = np.vstack((x,y))
                else:
                    lj_li = np.vstack((lj_li,x,y))

            # print('lj_li:', lj_li)
            # print("K_gains_mod:", np.shape(K_gains_mod))
            # take the last seen aptag from k_gains matrix
            K_gains_mod = np.delete(K_gains_mod,list(range(last_seen_aptag_id*2,last_seen_aptag_id*2+2)), axis = 1)
            u_3 += np.dot(K_gains_mod,lj_li)
            # print('u3',u_3)
            # print('u',u)
            u = u_1 + u_2 + u_3
            print('u',u)

            # publish control
            if len(u) != 0:    
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
    
    K_gains_path= shared_path + "K_gains_poly.csv"
    K_gains = read_matrix(K_gains_path)

    # K_added_path= shared_path + "K_added_new.csv"
    # K_added = read_matrix(K_added_path)

    jackal = Feedback_2D_Input(K_gains)#,K_added)
    jackal.get_rot_matrix_aptags()
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        jackal.compute_u()

    r.sleep()   




 

