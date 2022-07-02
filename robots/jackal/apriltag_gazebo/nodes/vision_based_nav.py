#!/usr/bin/env python

import rospy
import vision_based_navigation_ttt
import vision_based_navigation_ttt.nodes
from vision_based_navigation_ttt.nodes.controller import Controller
from vision_based_navigation_ttt.nodes.optical_flow import OFCalculator
from vision_based_navigation_ttt.nodes.tau_computation import TauComputationClass

# from sensor_msgs.msg import Image
from vision_based_navigation_ttt.msg import OpticalFlow
# from cv_bridge import CvBridgeError, CvBridge
# import cv2
# import sys
# import numpy as np

class navigation():
    def __init__(self):
        pass

    def optical_flow(self):
        # if len(sys.argv) < 2:
        #     parameter = str(0)
        #     # print("Parameter = 1, verbose mode")
        # else:
        #     parameter = sys.argv[1]
        rospy.init_node("optical_flow", anonymous=False)
        OFCalculator("1")
        # rospy.spin()

    def tau_computation(self):
        rospy.init_node("tau_computation", anonymous=False)
        TauComputationClass()
        # rospy.spin()


    def controller(self):
        rospy.init_node("controller", anonymous=False)
        Controller()
        # rospy.spin()

    def activate(self):
        self.optical_flow()
        self.tau_computation()
        self.controller()

    def deactivate(self):
        pass

if __name__=="__main__":
	# #initialise the node
	# rospy.init_node("", anonymous=True)
	navigation = navigation()
	#while the node is still on
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		navigation.activate()
		r.sleep()
