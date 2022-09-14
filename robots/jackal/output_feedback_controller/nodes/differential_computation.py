#!/usr/bin/env python3

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, Pose, TwistStamped
from apriltag_ros.msg import AprilTagDetectionArray
import transformation_utilities as tu
import numpy as np
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_geometry_msgs
from std_msgs.msg import Float64
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

class Input_Differential:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_linear_vel = rospy.Publisher('/linear_vel', Float64 ,queue_size=1)
        self.pub_vel = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        self.sub_u_input = rospy.Subscriber("/u_input",TwistStamped, self.feedback_control_callback)
        self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_callback)
        self.vel = Twist()
        self.aptag_transf = None
        self.selected_id = None
        self.u_input = None
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
            # '''If there's an AprilTag in the image'''
            min_distance = np.inf
            selected_apriltag = []
            for at in msg.detections:
                dist = np.linalg.norm([at.pose.pose.pose.position.x, at.pose.pose.pose.position.y, at.pose.pose.pose.position.z])
                if dist < min_distance:
                    min_distance = dist
                    self.selected_id = at.id[0]
                    selected_apriltag = at.pose.pose
            # print('id',self.selected_id )
            #change frame from camera to baselink
            source_frame = "front_realsense_gazebo"
            transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
            # pose_transformed = tf2_geometry_msgs.do_transform_pose(selected_apriltag, transform)
            # print('trans',tu.msg_to_se3(transform))
            ''' convert the stamped message '''
            self.aptag_transf =  np.dot(tu.msg_to_se3(transform),tu.msg_to_se3(at.pose.pose.pose))
            # print(self.aptag_transf)
        else:
            self.selected_id = None

    def to_tf(self,pos,ori):
        return np.block([[np.array(ori),pos.reshape((-1,1))],[0,0,0,1]])

    def feedback_control_callback(self,msg):
        if msg.twist:
            self.u_input = np.array([msg.twist.linear.x, msg.twist.linear.y])
            # print(self.u)
        else:
            self.u_input = None

# keep track of the unnormalized velocity when checking if it dropped less than a certain threshold
    def compute_input_parse(self):
        aptag_transf = self.aptag_transf
        selected_id = self.selected_id
        if aptag_transf is not None:
            ori = self.robot_pose(aptag_transf,selected_id)

            alpha = 0.2 # linear coef
            beta = 0.5 # angular coef

            linear_velocity = alpha * np.dot(self.u_input, ori[:2])
            print('Linear_Vel',linear_velocity)
            if linear_velocity < 0: 
                linear_velocity = 0
                angular_velocity = beta*np.cross(ori,self.u_input/np.linalg.norm(self.u_input))[2]
                
            else:
                angular_velocity = beta*np.cross(ori,self.u_input/np.linalg.norm(self.u_input))[2]

            self.vel.linear.x = linear_velocity/np.linalg.norm(self.u_input)
            self.vel.angular.z = angular_velocity 
           
            # print('linear',self.vel.linear.x)
            # print('angular',angular_velocity)
            print('reached')
            self.pub_vel.publish(self.vel)
            

## Get the orientation from different apriltags this only gets from the closest one
    def robot_pose(self,aptag_transf,selected_id):                           
        # orientation from landmark locations
        # print('1',self.position_landmark_inworld_matrix)
        # print("2",selected_id)
        # print('3',self.position_landmark_inworld_matrix[selected_id])
        # print('4',aptag_transf)
        ori_ldmark = np.dot(self.position_landmark_inworld_matrix[selected_id][:3,:3], \
            (aptag_transf[:3,:3]).T)
        # print('state', ori_ldmark)
        ori_ldmark = ori_ldmark[:3,0].flatten()
        ori_ldmark[2] = 0
        ori_ldmark /= np.linalg.norm(ori_ldmark)
     
        return ori_ldmark

if __name__ == "__main__":

    rospy.init_node("u_sparse_control")

    #  jackal = Jackal(K_gains)
    jackal = Input_Differential()
    # jackal.get_rot_matrix_aptags()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        jackal.compute_input_parse()
        r.sleep() 