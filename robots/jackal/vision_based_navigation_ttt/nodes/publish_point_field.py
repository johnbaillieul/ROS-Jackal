#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2,PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import numpy as np
import rospy
from vision_based_navigation_ttt.msg import TauComputation
import cv2
from sensor_msgs.msg import Image
import os
import pandas as pd
import time
from itertools import chain
from numpy import arctan2, sqrt
import numexpr as ne

class Velodyne:
    def __init__(self):
        # Velodyne Subscriber
        self.sub = rospy.Subscriber('/velodyne/points', PointCloud2, self.cloud_callback)
        self.vel_cloud_pub = rospy.Publisher('/points_cloud_', PointCloud2, queue_size=100)
        self.vel_rings_pub = rospy.Publisher('/points_rings_', PointCloud2, queue_size=100)
        # Tau Publisher
        # self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)

    def convertlist(self,longlist):
        tmp = list(chain.from_iterable(longlist))
        return np.array(tmp).reshape((len(longlist), len(longlist[0])))
    
    def cart2sph(self, x,y,z, ceval=ne.evaluate):
        """ x, y, z :  ndarray coordinates
            ceval: backend to use: 
                - eval :  pure Numpy
                - numexpr.evaluate:  Numexpr """
        azimuth = ceval('arctan2(y,x)')
        xy2 = ceval('x**2 + y**2')
        elevation = ceval('arctan2(z, sqrt(xy2))')
        r = eval('sqrt(xy2 + z**2)')
        return azimuth, elevation, r
    
    def cloud_callback(self, cloud):
        cloud_points = list(point_cloud2.read_points_list(cloud, field_names=("x", "y", "z"))) #, field_names=("x", "y", "z")
        # print('cloud', cloud_points)
        points_arr = self.convertlist(cloud_points) #np.asarray(cloud_points)
        # print('cloud', points_arr)
        indices = np.logical_and(points_arr[:,0] > 0, points_arr[:,2]> -0.28)
        points_arr = points_arr[indices]
        azimuth, _, _ = self.cart2sph(points_arr[:,0], points_arr[:,1], points_arr[:,2])
        print(np.max(azimuth),'max')
        print(np.min(azimuth),'min')
        # print('shape_points_arr',points_arr)
        # print('azimuth',azimuth)

        points = np.column_stack((points_arr, azimuth))
        print('points shpe', points)
        
        # print('size',np.size(points_arr_ ))
        ROI_el_ind = np.logical_and(points[:,3]>np.min(azimuth) , points[:,3]< np.min(azimuth)/2)
        ROI_el = points[ROI_el_ind]
        indices_azimuth = np.logical_and(points[:,3] > 0 , points[:,3]< np.max(azimuth)/2)
        points_arr_azimuth = points[indices_azimuth]
        # print('ring',points_arr_rings[:,0:3])
        self.publish_cloud(cloud, points_arr_azimuth[:,:3])


    def publish_cloud(self, cloud, points_arr_rings):
        # publish point cloud message
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
            ]

        header = Header()
        header.frame_id = cloud.header.frame_id
        header.stamp = cloud.header.stamp

        # pc2_cloud = point_cloud2.create_cloud(header, fields, cloud)
        # self.vel_cloud_pub.publish(pc2_cloud)

        pc2_rings = point_cloud2.create_cloud(header, fields, points_arr_rings)
        self.vel_rings_pub.publish(pc2_rings)

if __name__ == '__main__':
    rospy.init_node("tau_v", anonymous=False)
    vel = Velodyne()   
    rospy.spin()   
    
      
