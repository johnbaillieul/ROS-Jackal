#!/usr/bin/env python

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
import numpy as np
import tf2_ros
import tf.transformations 
import tf2_geometry_msgs



class Jackal:

	def __init__(self):
		self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
		self.sub_img_detec =  rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.Callback_detection)
		self.pose_x = None
		self.pose_y = None
		self.pose_z = None
		self.ang_pose_z= None
		self.spin = True
		self.move = False
		self.vel = Twist()
		self.prev_error = 0
		self.ap_id = [None]
		self.det_id = None
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		self.saved_time = rospy.Time.now()

	def Callback_detection(self,msg):
		if msg.detections and msg.detections[1].id not in self.ap_id:
			source_frame = "tag_" + str(msg.detections[1].id[0])
			transform = self.tfBuffer.lookup_transform("base_link", source_frame, rospy.Time(0), rospy.Duration(1.0))
			#print("before_trans",msg.detections[0].pose.pose)
			pose_transformed = tf2_geometry_msgs.do_transform_pose(msg.detections[1].pose.pose, transform)
			#print("pose_transf",pose_transformed)
			self.pose_x= pose_transformed.pose.position.x
			self.pose_y= pose_transformed.pose.position.y
			self.pose_z= pose_transformed.pose.position.z
			self.ang_pose_z= pose_transformed.pose.orientation.z
			self.move = True
			self.spin = False
			self.det_id = msg.detections[1].id
			print("spin",self.spin)
			print("id",self.det_id)


		else:
			self.pose_x = None
			self.pose_y = None
			self.pose_z = None
			self.ang_pose_z= None
			self.move = False
			self.spin = True
			print("spin",self.spin)


	def dist(self):
		return np.linalg.norm([self.pose_x, self.pose_y])

	def move_towards_tag(self):
		if self.pose_x is not None and self.pose_y is not None:
			dist_to_goal = self.dist() 
			print("dist", dist_to_goal)

if __name__=="__main__":
    
	#initialise the node
	rospy.init_node("ap_tag", anonymous=True)
	jack = Jackal()
	#while the node is still on
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		jack.move_towards_tag()
		r.sleep()