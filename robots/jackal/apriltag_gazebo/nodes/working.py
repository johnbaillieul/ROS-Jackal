#!/usr/bin/env python

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
import time
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
import numpy as np

class VelocityController:

    def __init__(self, Kp=3):
        '''
        '''
        self.Kp = Kp

    def proportional_control(self, error):
		return self.Kp * error


class Jackal:
	def __init__(self,prev_time = None, current_time= None):
		self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
		self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.Callback_detection)
		self.pose_x = None
		self.pose_y = None
		self.pose_z = None
		self.ang_pose_z= None
		self.pose_flag = False
		self.vel = Twist()
		self.controller = VelocityController()
		self.prev_error = 0
		self.ap_id = [None]
		self.det_id = None

        

	def Callback_detection(self,msg):
		if msg.detections and msg.detections[0].id not in self.ap_id :
			self.pose_x= msg.detections[0].pose.pose.pose.position.x
			self.pose_y= msg.detections[0].pose.pose.pose.position.y
			self.pose_z= msg.detections[0].pose.pose.pose.position.z
			self.ang_pose_z= msg.detections[0].pose.pose.pose.orientation.z
			self.pose_detected  = True
			self.det_id = msg.detections[0].id
			
			
		else:
			self.pose_x = None
			self.pose_y = None
			self.pose_z = None
			self.ang_pose_z= None
			self.pose_fdetected = False

	def spin(self):
		# adjust the velocity message
		self.vel.angular.z=1
		#publish it
		self.pub.publish(self.vel)

				
	def dist(self):
		return np.linalg.norm([self.pose_x, self.pose_y, self.pose_z])

	def move_towards_tag(self):
		if self.pose_x is not None and self.pose_y is not None and self.pose_z is not None:
			dist_to_goal = self.dist()
			while dist_to_goal > 0.65:
			
				current_error = self.ang_pose_z
				if current_error is None : 
					break
				self.vel.linear.x = 2
				self.vel.angular.z = self.controller.proportional_control(current_error)
				self.pub.publish(self.vel)

				if self.pose_x is not None and self.pose_y is not None and self.pose_z is not None:
					dist_to_goal = self.dist()
			
			self.update()

	def update(self):
		
		self.vel.linear.x = 0
		self.ap_id[0] = self.det_id

if __name__=="__main__":
    
	#initialise the node
	rospy.init_node("ap_tag", anonymous=True)
	jack = Jackal()
	#while the node is still on
	while not rospy.is_shutdown():
		if not jack.pose_flag:  #add rate.sleep()
			jack.spin()
		else:
			jack.move_towards_tag()

	

	