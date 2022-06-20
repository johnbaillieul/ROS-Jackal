#!/usr/bin/env python

#import the dependencies
import pid_controller  
import rospy
from geometry_msgs.msg import Twist
import time
import cv2
import numpy as np
from apriltag import apriltag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class AprilTag:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.sub =  rospy.Subscriber("/camera/color/raw_image", Image, self.imageColorCallback)

    def imageColorCallback(self, colordata):
        cv_image = self.bridge.imgmsg_to_cv2(colordata, desired_encoding='mono8')

    def detect_tag(self):
        #create the publisher object
        

        #define the rate of your publisher
        rate=rospy.Rate(10)

        #while the node is still on
        while True:
            # adjust the velocity message
            vel=Twist()
            vel.angular.z=1
            #publish it
            pub.publish(vel)
            #sleep to acheive the defined rate
            ####rate.sleep()

            #tag detection
            #sub to realsense2
          

            #image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

            detector = apriltag("tag36h11")
            detections = detector.detect(self.cv_image) 
            if len(detections) != 0: 
                break

    def move_to_tag(self):
        #create the publisher object
        pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)

        #define the rate of your publisher
        rate=rospy.Rate(10)

        #while the node is still on
        while True:

            # adjust the velocity message
            vel=Twist()
            ### add PID controller
            vel.angular.z=
            vel.linear.x=

            #publish it
            pub.publish(vel)

            #sleep to acheive the defined rate
            rate.sleep
    
#creat the function that will initialize the node
def ap_tag():

    #log the info to make sure the node was started
    rospy.loginfo("ap_tag node started ")
    #create the publisher object
    pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)

    #define the rate of your publisher
    rate=rospy.Rate(10)

    #define PID parameters
    kp = 0
    kd = 0 
    ki = 0

    #while the node is still on
    while True:

        # adjust the velocity message
        vel=Twist()

        vel.angular.z=1
        vel.linear.x=1

        #publish it
        pub.publish(vel)

        #sleep to acheive the defined rate
        rate.sleep

if __name__=="__main__":
    
    #initialise the node
    rospy.init_node("ap_tag", anonymous=True)

    #initiate object

    rospy.spin()
