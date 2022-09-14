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
        self.used_apriltags = [0,1,2,3,4,5,6,7,8,9,10,11] # add the apriltag ids that you used
        self.position_landmark_inworld_matrix = {}
        self.apriltag_dist = None
        
        # which cell do each landmark belong to 

        # self.K_to_cell = {0:'L1',1:'L6',2:'L6',3:'L3',4:'L1',5:'L1',6:'L2',7:'L2',8:'L3',9:'L3',10:'L3',11:'L4',12:'L5',13:'L5',14:'L4',15:'L2',16:'L4',17:'L6',18:'L5',19:'L6'}
        # self.cell_to_landmark = {'L1':[],'L2':[],'L3':[],'L4':[],'L5':[],'L6':[]}
        # self.current_cell = 0

        # self.cell_to_landmark = {'L1':[0,13,6,7,16], 'L2':[1,5,4,14],'L3':[11,2,10,15],'L4':[3,12,8,9]}
        # self.K_to_cell = {2:'L1',3:'L2',4:'L2',5:'L2',6:'L2',7:'L3',8:'L3',9:'L3',10:'L3',11:'L4',12:'L4',13:'L1',14:'L4'}
# d 0 x: 1.8907310962677002
# y: 5.275353908538818
# z: 0.5
# id 1 x: 7.329400062561035
# y: 2.7979321479797363
# z: 0.5
# id 2 x: 6.690731525421143
# y: -3.3704376220703125
# z: 0.5
# id 3 x: -3.1056039333343506
# y: -8.291247367858887
# z: 0.5
# id 4 x: -7.698666572570801
# y: -2.9030001163482666
# z: 0.5
# id 5 x: -4.860785007476807
# y: 4.497244358062744
# z: 0.5
# id 6 x: 1.398419976234436
# y: 2.067699909210205
# z: 0.5
# id 7 x: 3.3571999073028564
# y: 0.9101369976997375
# z: 0.5
# id 8 x: 2.626300096511841
# y: -2.404069662094116
# z: 0.5
# id 9 x: -1.8135725259780884
# y: -4.576581001281738
# z: 0.5
# id 10 x: -3.996692657470703
# y: -2.113091468811035
# z: 0.5
# id 11 x: -2.847109317779541
# y: 0.9576727151870728
# z: 0.5

    # def get_rot_matrix_aptags(self):
    #     rospy.wait_for_service('/gazebo/get_model_state')
    #     get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
    #     model = GetModelStateRequest()
    #     model.model_name = 'apriltag19'
    #     model.relative_entity_name = 'unit_box'
    #     result = get_model_srv(model)
    #     result_trans = tu.msg_to_se3(result.pose)
    #     # result_trans[3,2] = result_trans[3,2] + 0.5
    #     print('id',id,result.pose.position)
    #     print('getmodel',result_trans)
    #     # print(self.position_landmark_inworld_matrix)
        
    def get_rot_matrix_aptags(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        model = GetModelStateRequest()
        for id in self.used_apriltags:
            model.model_name = 'apriltag'+str(id)
            result = get_model_srv(model)
            print('id',id,result.pose.position)
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
            # pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
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
            if msg.data < threshold:
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
    
    K_gains_path= shared_path + "K_gains_new.csv"
    K_gains = read_matrix(K_gains_path)

    K_added_path= shared_path + "K_added_new.csv"
    K_added = read_matrix(K_added_path)

    jackal = Feedback_2D_Input(K_gains,K_added)
    jackal.get_rot_matrix_aptags()
   
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        jackal.compute_u()

    r.sleep()   




 

