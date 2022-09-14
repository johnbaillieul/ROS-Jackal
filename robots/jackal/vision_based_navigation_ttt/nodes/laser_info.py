#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
import sensor_msgs.msg

class laser_scan:
    def __init__(self):
        self.sub = rospy.Subscriber('/front/scan', LaserScan, self.callback)

    def callback(self, msg):
    #print(len(msg.ranges)) len is 2019 from 0-360
        ranges_array = len(msg.ranges) # number of beams
        # values at angles depend on the size of the range array and rangle range
        # ex  at min angle range[0] correspons to it and range[len(ranges)-1] corresponds to max angle]
       
        print('req_range',msg.ranges[ranges_array//2])
        
if __name__ == '__main__':
    rospy.init_node('revised_scan', anonymous=True)
    ls = laser_scan()
    rospy.spin()