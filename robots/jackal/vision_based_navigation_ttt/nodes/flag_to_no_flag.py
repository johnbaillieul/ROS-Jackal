#!/usr/bin/env python3
from tkinter import W
import numpy as np
import os
import pandas as pd
import xlsxwriter
from xlsxwriter import Workbook
import openpyxl

# path_tau = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/tau_values/tau_value"
# path_tau_no_flag = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/tau_values_no_flag/"

path_tau = os.environ["HOME"] + "/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/tau_values/tau_value"
path_tau_no_flag = os.environ["HOME"] + "/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/tau_values_no_flag/"

for i in range(1,5510):
    try:
        ps = openpyxl.load_workbook(path_tau + str(i) + '.xlsx')
        sheet = ps['Sheet1']
        tau_val = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]

        wb_tau =  xlsxwriter.Workbook(path_tau_no_flag + "tau_value" + str(i) + ".xlsx")
        wksheet_tau = wb_tau.add_worksheet()

        inf = 15
        if tau_val[0] != -1:
            wksheet_tau.write('A1',tau_val[0])
        else:
            wksheet_tau.write('A1',inf)
        if tau_val[1] != -1:
            wksheet_tau.write('B1',tau_val[1])
        else:
            wksheet_tau.write('B1',inf)
        if tau_val[2] != -1:
            wksheet_tau.write('C1',tau_val[2])
        else:
            wksheet_tau.write('C1',inf)
        if tau_val[3] != -1:
            wksheet_tau.write('D1',tau_val[3])
        else:
            wksheet_tau.write('D1',inf)
        if tau_val[4] != -1:
            wksheet_tau.write('E1',tau_val[4])
        else:
            wksheet_tau.write('E1',inf)
        wb_tau.close()
    except Exception as inst:
        print(i)    
        print(inst)