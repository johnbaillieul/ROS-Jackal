#!/usr/bin/env python

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
import numpy as np
import tf2_ros
import tf.transformations 
import tf2_geometry_msgs



class VelocityController:

    def __init__(self, Kp=0.2, Ki=0.5, Kd=0):
        '''
        '''
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def proportional_control(self, error):
		return self.Kp * error

    def integral_control(self, error, dt):
        return self.Ki * error * dt

    def derivative_control(self, error, previous_error, dt):
        return self.Kd * (error - previous_error)/dt




class Jackal:
	def __init__(self,prev_time = None, current_time= None):
		self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
		self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.Callback_detection)
		self.pose_x = None
		self.pose_y = None
		self.pose_z = None
		self.ang_pose_z= None
		self.spin = True
		self.move = False
		self.vel = Twist()
		self.controller = VelocityController()
		self.prev_error = 0
		self.ap_id = [None]
		self.det_id = None
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		self.saved_time = rospy.Time.now()

        

	def Callback_detection(self,msg):
		if msg.detections and msg.detections[0].id not in self.ap_id:
			transform = self.tfBuffer.lookup_transform("base_link", "front_realsense_gazebo" , rospy.Time(0), rospy.Duration(1.0))
			#print("before_trans",msg.detections[0].pose.pose)
			pose_transformed = tf2_geometry_msgs.do_transform_pose(msg.detections[0].pose.pose, transform)
			#print("pose_transf",pose_transformed)
			self.pose_x= pose_transformed.pose.position.x
			self.pose_y= pose_transformed.pose.position.y
			self.pose_z= pose_transformed.pose.position.z
			self.ang_pose_z= pose_transformed.pose.orientation.z
			self.move = True
			self.spin = False
			self.det_id = msg.detections[0].id
			print("spin",self.spin)
			
		else:
			self.pose_x = None
			self.pose_y = None
			self.pose_z = None
			self.ang_pose_z= None
			self.move = False
			self.spin = True
			print("spin",self.spin)

	def spinning(self):
		# adjust the velocity message
		self.vel.angular.z=1
		#publish it
		self.pub.publish(self.vel)
	
		
	def velocity_control(self, error, dt, prev_error):
		max_vel = 4 
		mv_p = self.controller.proportional_control(error)
		mv_i = self.controller.integral_control(error, dt)
		mv_d = self.controller.derivative_control(error, prev_error, dt)

		desired_vel = np.clip( mv_p + mv_i + mv_d, -max_vel, max_vel)
		return desired_vel

				
	def dist(self):
		return np.linalg.norm([self.pose_x, self.pose_y, self.pose_z])

	def move_towards_tag(self):
		if self.pose_x is not None and self.pose_y is not None and self.pose_z is not None:
			dist_to_goal = self.dist() 
			print("dist", dist_to_goal)
			current_error = self.ang_pose_z
	
			if dist_to_goal > 2.2 and current_error is not None:
				
				self.vel.linear.x = 1
				current_time = rospy.Time.now()
				dt = (current_time - self.saved_time).to_sec()

				self.vel.angular.z = self.velocity_control(current_error, dt, self.prev_error)
				self.saved_time = current_time

				self.pub.publish(self.vel)
				self.prev_error = current_error

				if self.pose_x is not None and self.pose_y is not None and self.pose_z is not None:
					dist_to_goal = self.dist()
			else:
				self.update()

	def update(self):
		self.move = False
		self.spin == True
		self.vel.linear.x = 0
		self.ap_id[0] = self.det_id
		self.prev_error = 0

if __name__=="__main__":
    
	#initialise the node
	rospy.init_node("ap_tag", anonymous=True)
	jack = Jackal()
	#while the node is still on
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		if jack.spin:
			jack.spinning()
		elif jack.move:
			jack.move_towards_tag()
		r.sleep()

	

	