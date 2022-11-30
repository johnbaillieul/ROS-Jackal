#!/usr/bin/env python3

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
import numpy as np
import tf2_ros
import tf.transformations 
import tf2_geometry_msgs
import transformation_utilities as tu

class Jackal:
	def __init__(self):
		self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
		self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.detection_callback)
			
		self.spin = True
		self.move = False
		self.vel = Twist()
		self.prev_error = 0
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		self.saved_time = rospy.Time.now()
		self.detected = False
		self.dist_to_tag = 0
		self.ap_in_baselink = None

		
		# source_frame = "front_realsense_gazebo"
		# 	# print(msg)
		# transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
		# self.transform_matrix = tu.msg_to_se3(transform)
		
	# def update_status(self):
	# 	# print("dist_to_tag", self.dist_to_tag)
	# 	if not self.detected :
	# 		self.move = False
	# 		self.spin = True 
	# 	elif self.detected and self.dist_to_tag < 1.5: # limit set baseed on trials
	# 		self.move = False
	# 		self.spin = True
	# 	else:
	# 		self.spin = False
	# 		self.move = True
			
	def on_apriltag_detection(self):
		if self.ap_in_baselink[0,3] is not None and self.ap_in_baselink[1,3] is not None and self.ap_in_baselink[2,3] is not None:
			self.dist_to_tag = self.get_dist() 

	def detection_callback(self,msg):
		if msg.detections:
			transform_gazebo_to_bl = self.tfBuffer.lookup_transform("base_link", "front_realsense_gazebo", rospy.Time(0), rospy.Duration(1.0))
			print('gatofro',tu.msg_to_se3(transform_gazebo_to_bl))
			# transform_cam_to_bl = self.tfBuffer.lookup_transform("base_link","front_realsense", rospy.Time(0), rospy.Duration(1.0))
			# transform_cam_to_bl = np.array([[1,0,0,-0.035],[0,1,0,0],[0,0,1,0.414],[0,0,0,1]])
			# print('frtobl',tu.msg_to_se3(transform_cam_to_bl))
			# self.ap_in_baselink = np.dot(transform_cam_to_bl,tu.msg_to_se3(transform_gazebo_to_cam))
			# np.array([[0, 0, 1, -0.035],[-1,0,0,0],[0, -1,  0,  0.414],[ 0,0,0,1]]) 
			# print('tra',self.ap_in_baselink)
			self.ap_in_baselink = np.dot(tu.msg_to_se3(transform_gazebo_to_bl), tu.msg_to_se3(msg.detections[0].pose.pose.pose))
			print('2',self.ap_in_baselink)
			self.on_apriltag_detection()
				
	def get_dist(self):
		dist = np.linalg.norm([self.ap_in_baselink[0,3], self.ap_in_baselink[1,3],self.ap_in_baselink[2,3]])
		# dist2 = np.linalg.norm([self.ap_in_baselink[0,3], self.ap_in_baselink[1,3]])
		print('dist',dist)
		return dist

	# def move_towards_tag(self):
	# 	current_error = self.ap_in_baselink[1,3]
	# 	print("current_error",current_error)
	# 	if current_error is not None:
	# 		self.vel.linear.x = 1
	# 		current_time = rospy.Time.now()
	# 		dt = (current_time - self.saved_time).to_sec()
	# 		pid_output = self.velocity_control(current_error, dt, self.prev_error)
	# 		# print("pid_output ", pid_output)
	# 		self.vel.angular.z = pid_output 
	# 		self.saved_time = current_time

	# 		self.pub.publish(self.vel)
	# 		self.prev_error = current_error


if __name__=="__main__":
    
	#initialise the node
	rospy.init_node("ap_tag", anonymous=True)
	jack = Jackal()
	#while the node is still on
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		r.sleep()

	

	