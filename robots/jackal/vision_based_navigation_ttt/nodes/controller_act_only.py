#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from vision_based_navigation_ttt.msg import TauComputation
import numpy as np
import time
import sys
from subprocess import call


# Function to check if there is an obstacle in the center part of the image
def find_obstacle(self, mean, t):

	if self.center and (mean <= t):
		self.obstacle = True
	else:
		self.obstacle = False

# Function to set a threshold for TTT values
def threshold(value, limit):
	if value >= limit:
		value = limit
	elif value <= -limit:
		value = -limit
	return value

# Function to compute the value of the control action
def perceive(self):
	if self.right and self.left:
		self.tau_diff = self.mean_tau_l - self.mean_tau_r
	if self.extreme_left and self.extreme_right:
		self.tau_diff_extreme = self.mean_tau_el - self.mean_tau_er
	if self.right and self.extreme_right:
		self.diff_right = self.right - self.extreme_right
	if self.left and self. extreme_left:
		self.diff_left = self.left - self.extreme_left


class Controller:

	def __init__(self):

		######## IMPORTANT PARAMETERS: ########
		# Minimum number of features needed to compute the average TTT for each ROI
		# Percentage of discarded TTT values for each ROI
		self.percentage = 0.25
		# Saturation values for the control input
		self.max_u = 1
		self.max_control_diff = 0.5

		self.robot_publisher = "jackal_velocity_controller/cmd_vel"
		########################################

		# Tau Data Subscriber
		self.tau_values = rospy.Subscriber("tau_values", TauComputation, self.callback)

		# Variables initialization
		# Sense and Act cycles variables
		self.act_rate = 50

		self.double_act_action = False

		# Boot parameters
		self.init_cnt = 0
		self.max_init = 10

		# Control parameters initialization
		# Variables to understand the geometry of the environment
		self.obstacle = False
		self.extreme_right = True
		self.extreme_left = True
		self.right = True
		self.left = True
		self.center = True
		# Time before which the robot must start to turn if there is a corner 
		self.time_to_turn = 3
		# Time before which the robot must start to turn if there is an obstacle
		self.time_to_obstacle = 2
		# Initialization input controls
		self.u = 0
		self.u_e = 0
		
		# Initialization average TTT values for each ROI 
		self.mean_tau_er = 0
		self.mean_tau_el = 0
		self.mean_tau_r = 0
		self.mean_tau_l = 0
		self.mean_tau_center = 0

		# Single wall strategy
		self.tau_diff_max = True
		# Variables initialization
		self.first_tdm_r = True
		self.first_tdm_l = True
		# Variable 
		self.actual_wall_distance = 0
		self.actual_wall_distance_e = 0
		self.tau_diff = 0
		self.tau_diff_extreme = 0
		self.diff_left = 0
		self.diff_right = 0
		self.prev_diff_r = 0
		self.prev_diff_l = 0
		self.curr_diff_r = 0
		self.curr_diff_l = 0
		self.dist_from_wall_er = 0
		self.dist_from_wall_el = 0
		self.dist_from_wall_r = 0
		self.dist_from_wall_l = 0
		self.safe_dist = 0.5
		self.prev_controls = np.array([])
		self.control = 0
		self.constant_left = 1
		self.constant_right = 1
		self.lin_vel = 1

		# Steering signal Publisher Jackal
		self.steering_signal = rospy.Publisher(self.robot_publisher, Twist, queue_size=10)

	def callback(self, data):
		# Boot phase performs only in the first sense phase to allow the stabilization of the OF values
		if self.init_cnt < self.max_init:
			msg = Twist()
			msg.linear.x = float(self.lin_vel)
			msg.angular.z = float(0)
			self.steering_signal.publish(msg)
			self.init_cnt += 1
			return

		# Go straight
		msg = Twist()
		msg.linear.x = float(self.lin_vel)
		msg.angular.z = float(0)
		self.steering_signal.publish(msg)

		# Data acquisition from TauValues topic
		final_tau_right_e = data.tau_er
		final_tau_left_e = data.tau_el
		final_tau_right = data.tau_r
		final_tau_left = data.tau_l
		final_tau_center = data.tau_c
		
		
		# check if tau value is valid or not
		if final_tau_right_e > 0:
			self.mean_tau_er = final_tau_right_e
		else:
			self.extreme_right = False # no TTT value in that ROI

		if final_tau_left_e > 0:
			self.mean_tau_el = final_tau_left_e
		else:
			self.extreme_left = False

		if final_tau_center > 0:
			self.mean_tau_c = final_tau_center
		else:
			self.center = False

		if final_tau_right > 0:
			self.mean_tau_r = final_tau_right
		else:
			self.right = False

		if final_tau_left > 0:
			self.mean_tau_l = final_tau_left
		else:
			self.left = False

		perceive(self)

		# Set values for Single Wall strategy by using as c the TTT value at t=0
		self.dist_from_wall_er = self.mean_tau_er
		self.dist_from_wall_el = self.mean_tau_el
		self.dist_from_wall_r = self.mean_tau_r
		self.dist_from_wall_l = self.mean_tau_l

		self.double_act_action = True
		
		# Act Phase
	
		r_act = rospy.Rate(self.act_rate)

		# To understand if there is a wall in front of the robot
		find_obstacle(self, self.mean_tau_center, self.time_to_turn)
		# Set gain
		self.kp = 0.1
		self.kp_e = 0.2
		self.kd = 0.5

		# Initialization of the control inputs
		control_e = 0
		control_m = 0
		control = 0
		self.tau_diff_max = True

		# If both extreme ROIs has an average TTT values ---> use tau_balancing
		if self.extreme_left and self.extreme_right:
			control_e = self.tau_diff_extreme
			self.tau_diff_max = False
		# If both lateral ROIs has an average TTT values ---> use tau_balancing
		if self.left and self.right:
			control_m = self.tau_diff
			self.tau_diff_max = False

		# If tau balancing must be used
		if not self.tau_diff_max:
			if self.obstacle:
				find_obstacle(self, self.mean_tau_center, self.time_to_obstacle)	# The obstacle is at 2 second?
				if self.extreme_left and self.extreme_right:
					if self.obstacle and (abs(self.tau_diff_extreme) < 0.5):	  # The obstacle is an object
																					# in the middle of the path
						self.kp_e = 1.4
						if self.double_act_action:
							self.double_act_action = False
							
						if self.mean_tau_el > self.mean_tau_er:
							control = self.kp_e * (self.mean_tau_el - self.constant_left)
							print('\033[1m'+ "Obstacle! Go left" + '\033[0m')
							print("Control: " + str(control))
						else:
							control = -self.kp_e * (self.mean_tau_er - self.constant_right)
							print('\033[1m' + "Obstacle! Go right" + '\033[0m')
							print("Control: " + str(control))
					else:														 # The obstacle is a wall
																					# belonging to a turn
						self.kp = 1
						self.kp_e = 1.3
						control = self.kp_e * control_e + self.kp * control_m
						print('\033[1m'+"Turn ahead"+'\033[0m')
						print("Diff Extreme: " + str(self.tau_diff_extreme))
						print("Diff Medium: " + str(self.tau_diff))
						print("Control: " + str(control))
				else:
					if self.obstacle and (abs(self.tau_diff) < 0.5):
						self.kp_e = 1.2
						if self.double_act_action:
							self.double_act_action = False
							
						if self.mean_tau_l > self.mean_tau_r:
							control = self.kp_e * (self.mean_tau_l - self.constant_left)
							print('\033[1m'+"Obstacle! Go left (medium)"+'\033[0m')
							print("Control: " + str(control))
						else:
							control = -self.kp_e * (self.mean_tau_r - self.constant_right)
							print('\033[1m'+"Obstacle! Go right (medium)"+'\033[0m')
							print("Control: " + str(control))
					else:
						self.kp = 1.5
						print('\033[1m'+"Turn ahead"+'\033[0m')
						print("Diff Medium with no extreme: " + str(self.tau_diff))
						control = self.kp_e * control_e + self.kp * control_m
						print("Control: " + str(control))
			else:																	   # no obstacles
				if np.size(self.prev_controls) == 2:
					control_diff = self.prev_controls[1] - self.prev_controls[0]
					u_diff = threshold((self.kd * control_diff), self.max_control_diff)
				else:
					u_diff = 0
				u_prop = self.kp_e * control_e + self.kp * control_m
				if u_diff * u_prop <= 0:
					control = u_prop + u_diff
					print('\033[1m'+"Tau Balancing"+'\033[0m')
					print("control tau balancing: " + str(control))
				else:
					control = u_prop
					print('\033[1m'+"Tau Balancing"+'\033[0m')
					print("control tau balancing: " + str(control))

			self.control = control
			self.first_tdm_r = True
			self.first_tdm_l = True

		elif self.extreme_right:   # Only extreme right ROI has a TTT value ---> Single Wall strategy
			self.kp = 1
			print('\033[1m'+"Single wall strategy on extreme right"+'\033[0m')
			if self.first_tdm_r:
				self.first_tdm_r = False
				self.first_tdm_l = True
				self.actual_wall_distance = self.dist_from_wall_r + self.safe_dist
				self.actual_wall_distance_e = self.dist_from_wall_er + self.safe_dist
				self.actual_wall_distance = 1
				self.actual_wall_distance_e = 1
			control = -self.kp * (self.mean_tau_er - self.actual_wall_distance_e)
			print("actual distance: " + str(self.actual_wall_distance_e))
			print("control: " + str(control))
		elif self.right:			# Only right ROI has a TTT value ---> Single Wall strategy
			self.kp = 1
			print('\033[1m'+"Single wall strategy on right"+'\033[0m')
			if self.first_tdm_r:
				self.first_tdm_r = False
				self.first_tdm_l = True
				self.actual_wall_distance = self.dist_from_wall_r + self.safe_dist
				self.actual_wall_distance_e = self.dist_from_wall_er + self.safe_dist
				self.actual_wall_distance = 1
				self.actual_wall_distance_e = 1
			control = -self.kp * (self.mean_tau_r - self.actual_wall_distance)
			print("actual distance: " + str(self.actual_wall_distance))
			print("control: " + str(control))
		elif self.extreme_left:		 # Only extreme left ROI has a TTT value ---> Single Wall strategy
			self.kp = 1
			if self.first_tdm_l:
				self.first_tdm_l = False
				self.first_tdm_r = True
				self.actual_wall_distance = self.dist_from_wall_l + self.safe_dist
				self.actual_wall_distance_e = self.dist_from_wall_el + self.safe_dist
				self.actual_wall_distance = 1
				self.actual_wall_distance_e = 1
			print('\033[1m'+"Single wall strategy on extreme left"+'\033[0m')
			control = self.kp * (self.mean_tau_el - self.actual_wall_distance_e)
			print("actual distance: " + str(self.actual_wall_distance_e))
			print("control: " + str(control))
		elif self.left:				 # Only left ROI has a TTT value ---> Single Wall strategy
			self.kp = 1
			print('\033[1m'+"Single wall strategy on left"+'\033[0m')
			if self.first_tdm_l:
				self.first_tdm_l = False
				self.first_tdm_r = True
				self.actual_wall_distance = self.dist_from_wall_l + self.safe_dist
				self.actual_wall_distance_e = self.dist_from_wall_el + self.safe_dist
				self.actual_wall_distance = 1
				self.actual_wall_distance_e = 1
			control = self.kp * (self.mean_tau_l - self.actual_wall_distance)
			print("actual distance: " + str(self.actual_wall_distance))
			print("control: " + str(control))

		# Verify the value of the control action and limit it if needed
		control = threshold(control, self.max_u)

		# Publish Steering signal
		msg = Twist()
		msg.linear.x = float(self.lin_vel)
		msg.angular.z = float(control)

		self.steering_signal.publish(msg)
		r_act.sleep()
			


def controller():
	rospy.init_node("controller", anonymous=False)
	Controller()
	rospy.spin()


if __name__ == '__main__':
	controller()
