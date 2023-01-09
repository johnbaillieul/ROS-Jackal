#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2,PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge,  CvBridgeError 
import numpy as np
from tkinter import W
import rospy
from vision_based_navigation_ttt.msg import TauComputation
import cv2
from sensor_msgs.msg import Image
import os
import pandas as pd
import xlsxwriter
from xlsxwriter import Workbook
import time
from itertools import chain

class TauComp:
    def __init__(self):
        
        self.pointcloud_sub = rospy.Subscriber('/velodyne/points', PointCloud2, self.cloud_callback)
        self.pointcloud_pub = rospy.Publisher('/points_cloud_new', PointCloud2, queue_size=100)
       
    def convertlist(self,longlist):
        tmp = list(chain.from_iterable(longlist))
        return np.array(tmp).reshape((len(longlist), len(longlist[0])))

    def cloud_(self, cloud, points_arr_):
        # start = time.time()
        cloud_points = list(point_cloud2.read_points_list(cloud, skip_nans=True, field_names = ("x", "y", "z")))
        # start = time.time()
        points_arr= self.convertlist(cloud_points) #np.asarray(cloud_points)
        # print(f"list_to_array conv:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

        indices = np.logical_and(points_arr[:,0] > 0, points_arr[:,2]> 1)
        points_arr_ = points_arr[indices]
        print('size',np.size(points_arr_ ))
       
        return points_arr_
            
    def cloud_callback(self, data):
        start = time.time()
        points_arr_ = None
        
        if data is not None:
            # start = time.time()
            points_arr_ = self.cloud_(data, points_arr_)
            self.publish_cloud(data, points_arr_)
        print(f"wholefunc:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

    def publish_cloud(self, cloud, points_arr):
        # publish point cloud message
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
            ]

        header = Header()
        header.frame_id = cloud.header.frame_id
        header.stamp = cloud.header.stamp

        pc2 = point_cloud2.create_cloud(header, fields,points_arr)
        self.pointcloud_pub.publish(pc2)


if __name__ == '__main__':
    rospy.init_node("tau_v", anonymous=False)
    tau = TauComp()   
    rospy.spin()   
    
      