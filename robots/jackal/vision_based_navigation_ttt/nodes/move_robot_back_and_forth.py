#!/usr/bin/env python3

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan 
import numpy as np

class Jackal:
    def __init__(self):
        self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        self.vel = Twist()
        # Lidar Subscriber
        self.sub = rospy.Subscriber('/front/scan', LaserScan, self.callback)
        self.dis_at_90 = None
        self.duration = rospy.Duration(0.25)

    def callback(self, msg):
        # print(msg.ranges[len(msg.ranges)//2])
        self.dis_at_90 = msg.ranges[len(msg.ranges)//2]
        print(self.dis_at_90)
        if self.dis_at_90 > 0.3:
            self.vel.linear.x = 0.5
            #publish it
            self.pub.publish(self.vel)
        else: 
            start_time = rospy.Time.now()
            print('s',start_time)
            print('s3',start_time + self.duration)
            while rospy.Time.now() < start_time + self.duration:
                # print('n',rospy.Time.now())
                self.vel.linear.x = -0.5
                self.pub.publish(self.vel)

if __name__=="__main__":
	#initialise the node
	rospy.init_node("ap_tag", anonymous=True)
	jack = Jackal()
	#while the node is still on
	r = rospy.Rate(10)
	rospy.spin()

	

	