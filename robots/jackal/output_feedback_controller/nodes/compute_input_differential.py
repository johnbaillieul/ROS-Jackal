#!/usr/bin/env python

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, Pose, TwistStamped
from apriltag_ros.msg import AprilTagDetectionArray
import transformation_utilities as tu
import numpy as np
import tf2_ros
import tf.transformations 
import tf2_geometry_msgs
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry


class Input_Differential:
    def __init__(self):
        self.odom_transf = None
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_linear_vel = rospy.Publisher('/linear_vel', Float64 ,queue_size=1)
        self.pub_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        self.sub_u_input = rospy.Subscriber("/u_input",TwistStamped, self.feedback_control_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.sub_odom =  rospy.Subscriber("/my_odom", Odometry, self.odom_callback)
        self.aptag_transf = None
        self.selected_id = None
        self.u_input = None
        self.position_landmark_inworld_matrix = {}
        self.position_landmark_inworld = {1:[-8.24439, -5.8941, 0.5, 0, 0, 0, 1.0],
                                          2:[-8.27337, -1.68115, 0.5, 0, 0, 0, 1.0],
                                          3:[-8.23126, 1.44051, 0.5, 0, 0, 0, 1.0],
                                          4:[-5.65321, 3.03627, 0.5, 0, 0, -0.717185376228,0.696882440678],
                                          5:[-2.52459, 3.22243, 0.5, 0, 0,  -0.699616049545, 0.714518987305],
                                          6:[1.31312, -2.15597, 0.5, 0, 0, -0.999693206373, 0.0247687935147],
                                          7:[1.45869, -6.20846, 0.5, 0, 0,0.99999418143,0.00341132017855],
                                          8:[-2.46765, -8.80024, 0.5, 0, 0, 0.705986429932,0.708225360145],
                                          9:[-5.70343, -8.65199, 0.5, 0, 0, 0.692161786123,0.721742379129],
                                          11:[-4.10288, -5.19704, 0.5, 0, 0, 0.999988854557, 0.00472130925446],
                                          12:[-4.10288, -2.02095, 0.5, 0, 0,-0.999944861574, 0.0105011337853],
                                          13:[-1.98548, -2.07486, 0.5, 0, 0, 0, 1.0],
                                          14:[-1.93215, -4.18616, 0.5, 0, 0, 0, 1.0],
                                          15:[1.22551, 1.92208, 0.5, 0, 0, 0.999930365151,0.0118010528658]
                                         }
    
    def apriltag_callback(self,msg):

        if msg.detections:
            # print(msg.detections)
            # '''If there's an AprilTag in the image'''
            min_distance = np.inf
            selected_apriltag = []
            for at in msg.detections:
                dist = np.linalg.norm([at.pose.pose.pose.position.x, at.pose.pose.pose.position.y, at.pose.pose.pose.position.z])
                if dist < min_distance:
                    min_distance = dist
                    self.selected_id = at.id[0]
                    selected_apriltag = at.pose.pose

                #change frame from camera to baselink
                source_frame = "front_realsense_gazebo"
                transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
                pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
                ''' convert the stamped message '''
                self.aptag_transf = tu.msg_to_se3(pose_transformed.pose)
                # print(self.at_transf)
        else:
            self.selected_id = None

    def qt_to_matrix(self): 
        # converts the position of the apriltags from quaternions to transformation matrices
        for id in self.position_landmark_inworld.keys():
                pose = Pose()
                pose.position.x = self.position_landmark_inworld[id][0]
                pose.position.y = self.position_landmark_inworld[id][1]
                pose.position.z = self.position_landmark_inworld[id][2]
                pose.orientation.x = self.position_landmark_inworld[id][3]
                pose.orientation.y = self.position_landmark_inworld[id][4]
                pose.orientation.z = self.position_landmark_inworld[id][5]
                pose.orientation.w = self.position_landmark_inworld[id][6]

                self.position_landmark_inworld_matrix[id] = tu.msg_to_se3(pose)


    def to_tf(self,pos,ori):
        return np.block([[np.array(ori),pos.reshape((-1,1))],[0,0,0,1]])

    def feedback_control_callback(self,msg):
        if msg.twist:
            self.u_input = np.array([msg.twist.linear.x, msg.twist.linear.y])
            # print(self.u)
        else:
            self.u_input = None

    def odom_callback(self,msg):
        if msg:
            # for debugging
            self.odom_transf = tu.msg_to_se3(msg.pose.pose)
         
    def compute_input_parse(self):
        aptag_transf = self.aptag_transf
        selected_id = self.selected_id
        ori = self.robot_pose(aptag_transf,selected_id)
    
        if self.u_input is not None and aptag_transf is not None and ori is not None:
    
            alpha = 0.2 
            beta = 0.7

            linear_velocity = alpha * np.dot(self.u_input, ori[:2])/np.linalg.norm(self.u_input)
            angular_velocity = beta*np.cross(ori,self.u_input/np.linalg.norm(self.u_input))[2]
          
            self.vel.angular.z = angular_velocity 
            self.vel.linear.x = linear_velocity

            self.pub_vel.publish(self.vel)
            self.pub_linear_vel.publish(linear_velocity)

## Get the orientation from different apriltags this only gets from the closest one
    def robot_pose(self,aptag_transf,selected_id):
        if self.odom_transf is not None:
            
            # ground truth from odom info
            ori_odom = self.odom_transf[:3,0].flatten()     
            ori_odom[2] = 0
            ori_odom /= np.linalg.norm(ori_odom)
            # print('ori_odom', ori_odom)

            # orientation from landmark locations
            ori_ldmark = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3], np.linalg.inv(aptag_transf[:3,:3]))
            print('state', ori_ldmark)
            ori_ldmark = ori_ldmark[:3,0].flatten()
            ori_ldmark[2] = 0
            ori_ldmark /= np.linalg.norm(ori_ldmark)
            # print('ori_state', ori_state)
            return ori_ldmark

if __name__ == "__main__":

    rospy.init_node("u_sparse_control")

    #  jackal = Jackal(K_gains)
    jackal = Input_Differential()
    jackal.qt_to_matrix()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        jackal.compute_input_parse()
	r.sleep() 