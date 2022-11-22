#!/usr/bin/env python3
from tkinter import W
import rospy
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt.msg import TauComputation
from cv_bridge import CvBridgeError, CvBridge
import cv2
from sensor_msgs.msg import Image 
import json
import os
import pandas as pd
import xlsxwriter
from xlsxwriter import Workbook
from PIL import Image as im

class collect_data():
    def __init__(self):
        # Lidar Subscriber
        self.sub = rospy.Subscriber('/front/scan', LaserScan, self.callback)
        # Tau Publisher
        self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)
        # Raw Image Subscriber
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        self.ranges = None
        self.increments = None
        self.angle_min = None
        self.angle_max = None
        # Initialize Image acquisition
        self.bridge = CvBridge()
        self.get_variables()
        self.count_2 += 1
        # self.update_variables()

        self.curr_image = None
        self.path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/training_images/"
        vel = '0.5' # velocity 
        self.folder_name = 'training_images_' + str(self.count_2) + '_v_' + vel + '/'
        os.mkdir(self.path_folder + self.folder_name)    
       

    def callback_img(self, data):
        try:
            print("imagesfun")
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.curr_image = im.fromarray(self.curr_image)
    
        except CvBridgeError as e:
            print(e)
            return
      
    def callback(self, msg):
            start_ind  = 230 #0
            end_ind = 488 #len(msg.ranges) - 1   #488 #
            # print('ol',msg.angle_max)
            self.angle_min = msg.angle_min + start_ind * msg.angle_increment
            self.angle_max = msg.angle_min + end_ind * msg.angle_increment
            self.increments = msg.angle_increment
            self.ranges = msg.ranges[230:489]

    def save_image(self, count : int, shared_path): 
            img_name= str(count) + '.png'
            path = shared_path + img_name
            picture = self.curr_image.save(path)

    def get_variables(self):
        # print("get_variables")
        path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/"
        file = open(path + "variables.json")
        data = json.load(file)
        file.close()
        self.count = data["count"]
        self.count_2 = data["count_2"]
        
    def update_variables(self):
        # print("update_variables")
        path = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/"
        file = open(path + "variables.json", "w")
        updated_data = {"count": self.count, "count_2": self.count_2}
        json.dump(updated_data, file)
        file.close()
    
    def get_tau_values(self):
        print("ppp")
        # Start by opening the spreadsheet and selecting the main sheet
        path_tau = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/tau_values/"
        # create_folder
        path_images = self.path_folder + self.folder_name
        wb_tau =  xlsxwriter.Workbook(path_tau + "tau_value" + str(self.count) + ".xlsx")
        wksheet_tau = wb_tau.add_worksheet()
       
        if self.curr_image is not None:
            curr_image = self.curr_image
            if self.ranges is not None:
                theta_rd = np.arange(self.angle_min, self.angle_max + self.increments, self.increments, dtype=float) # generated within the half-open interval [start, stop).
                theta_deg = theta_rd * (180/np.pi)
                ranges = self.ranges
                tau_val = np.array([])
                for i in range(len(ranges)):
                    tau_val = np.append(tau_val,abs(ranges[i]*np.cos(theta_deg[i])))
                self.tau_val = tau_val
                                
                # Extreme left
                count_inf_el = 0
                count_el = 0
                tau_val_el = self.tau_val[204:259]
                tau_el = 0
                for i in range(len(tau_val_el)):
                    if tau_val_el[i] == np.inf: 
                        count_inf_el += 1
                        if count_inf_el > int(len(tau_val_el)/2):
                            tau_el = np.inf
                            break
                    else:
                        tau_el += tau_val_el[i]
                        count_el += 1
                if count_el ==0:
                    tau_el = np.inf
                else:
                    tau_el = tau_el / count_el

                # Extreme right
                count_inf_er = 0
                count_er = 0
                tau_val_er = tau_val[0:56]
                tau_er = 0       
                for i in range(len(tau_val_er)):
                    if tau_val_er[i] == np.inf: 
                        count_inf_er += 1
                        if count_inf_er > int(len(tau_val_er)/2):
                            tau_er = np.inf
                            break
                    else: 
                        tau_er += tau_val_er[i]
                        count_er += 1
                if count_er == 0:
                    tau_er = np.inf
                else:
                    tau_er = tau_er / count_er

                # left
                count_inf_l = 0
                count_l = 0
                tau_val_l = tau_val[154:205]
                tau_l = 0
                for i in range(len(tau_val_l)):
                    if tau_val_l[i] == np.inf: 
                        count_inf_l += 1
                        if count_inf_l > int(len(tau_val_l)/2):
                            tau_l = np.inf
                            break
                    else: 
                        tau_l += tau_val_l[i]
                        count_l += 1
                if count_l == 0:
                    tau_l = np.inf
                else:
                    tau_l = tau_l / count_l

                # Right
                count_inf_r = 0
                count_r = 0
                tau_val_r = tau_val[55:103]
                tau_r = 0
                for i in range(len(tau_val_r)):
                    if tau_val_r[i] == np.inf: 
                        count_inf_r += 1
                        if count_inf_r > int(len(tau_val_r)/2):
                            tau_r = np.inf
                            break
                    else: 
                        tau_r += tau_val_r[i]
                        count_r += 1
                if count_r == 0:
                    tau_r = np.inf
                else:
                    tau_r = tau_r / count_r

                # Centre
                count_inf_c = 0
                count_c = 0
                tau_val_c = tau_val[117:144]
                tau_c = 0
                for i in range(len(tau_val_c)):
                    if tau_val_c[i] == np.inf: 
                        count_inf_c+=1
                        if count_inf_c > int(len(tau_val_c)/2):
                            tau_c = np.inf
                            break
                    else: 
                        tau_c += tau_val_c[i]
                        count_c +=1
                if count_c == 0:
                    tau_c = np.inf
                else:
                    tau_c = tau_c / count_c 
                tau_val = [tau_el, tau_l, tau_c, tau_r, tau_er]

            
                # self.save_image(self.count, path_images)
                # inf = -1
                # try:
                #     wksheet_tau.write('A1',tau_val[0])
                # except:
                #     wksheet_tau.write('A1',inf)
                # try:
                #     wksheet_tau.write('B1',tau_val[1])
                # except:
                #     wksheet_tau.write('B1',inf)
                # try:
                #     wksheet_tau.write('C1',tau_val[2])
                # except:
                #     wksheet_tau.write('C1',inf)
                # try:
                #     wksheet_tau.write('D1',tau_val[3])
                # except:
                #     wksheet_tau.write('D1',inf)
                # try:
                #     wksheet_tau.write('E1',tau_val[4])
                # except:
                #     wksheet_tau.write('E1',inf)
                # wb_tau.close()

                self.count += 1
                print(self.count)
                # self.update_variables()

if __name__ == "__main__":
    rospy.init_node("collect_data")
    collect = collect_data()
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        collect.get_tau_values()
        r.sleep()   
    
