import os
import openpyxl
import numpy as np
import xlsxwriter
# retrive the direction from the filename
count =1
path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/cnn_lidar_error/"
folder_name = str(count) 
os.mkdir(path_folder + folder_name) 
# path_tau = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/tau_values/"   
# path_tau_no_flag = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/tau_values_no_flag/"   
# y_2 = []
# for i in range(1,4):
#     ps = openpyxl.load_workbook(path_tau + "tau_value" + str(i) + ".xlsx")
#     sheet = ps['Sheet1']
#     tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
#     y_temp = []
#     for i in tau_values:
#         if i == -1:
#             y_temp.append(0)
#         else:
#             y_temp.append(1)
#     y_2.append(y_temp)
# print(y_2)
# y_2 = np.asarray(y_2)
# print(y_2)

    # wb_tau =  xlsxwriter.Workbook(path_tau_no_flag + "tau_value" + str(i) + ".xlsx")
    # wksheet_tau = wb_tau.add_worksheet()



    # limit = 10
    # if sheet['A1'].value == -1:
    #     wksheet_tau.write('A1', limit)
    # else: 
    #     wksheet_tau.write('A1', sheet['A1'].value)
    
    # if sheet['B1'].value == -1:
    #     wksheet_tau.write('B1', limit)
    # else: 
    #     wksheet_tau.write('B1', sheet['B1'].value)

    # if sheet['C1'].value == -1:
    #     wksheet_tau.write('C1', limit)
    # else: 
    #     wksheet_tau.write('C1', sheet['C1'].value)

    # if sheet['D1'].value == -1:
    #     wksheet_tau.write('D1', limit)
    # else: 
    #     wksheet_tau.write('D1', sheet['D1'].value)

    # if sheet['E1'].value == -1:
    #     wksheet_tau.write('E1', limit)
    # else: 
    #     wksheet_tau.write('E1', sheet['E1'].value)

    # wb_tau.close()
