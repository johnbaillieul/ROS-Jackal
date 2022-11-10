#! /usr/bin/env python3
from ftplib import error_perm
import rospy
import numpy as np
import os
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd

count = 5
filter = 'mad_range2_'
# data = pd.read_excel(path_tau + 'er_error_array.xlsx')
path_er = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/er_error_images/"
book_er = xlrd.open_workbook(path_er + 'er_error_array_' + filter  + str(count) + '.xlsx')
sheet_er = book_er.sheet_by_name('Sheet1')
er_error_array = [sheet_er.cell_value(r, 1) for r in range(sheet_er.nrows)]
# print(er_error_array)
er_error_array = [x for x in er_error_array  if np.abs(x)<10]  


path_el = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/el_error_images/"
book_el = xlrd.open_workbook(path_el + 'el_error_array_' + filter  + str(count) + '.xlsx')
sheet_el = book_el.sheet_by_name('Sheet1')
el_error_array = [sheet_el.cell_value(r, 1) for r in range(sheet_el.nrows)]
# print(er_error_array)
el_error_array = [x for x in el_error_array  if np.abs(x)<10]  


path_l = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/l_error_images/"
book_l = xlrd.open_workbook(path_l + 'l_error_array_' + filter + str(count) + '.xlsx')
sheet_l = book_l.sheet_by_name('Sheet1')
l_error_array = [sheet_l.cell_value(r, 1) for r in range(sheet_l.nrows)]
# print(er_error_array)
l_error_array = [x for x in l_error_array  if np.abs(x)<10]  

path_r = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/r_error_images/"
book_r = xlrd.open_workbook(path_r + 'r_error_array_' + filter + str(count) + '.xlsx')
sheet_r = book_r.sheet_by_name('Sheet1')
r_error_array = [sheet_r.cell_value(r, 1) for r in range(sheet_r.nrows)]
# print(er_error_array)
r_error_array = [x for x in r_error_array  if np.abs(x)<10]  

path_c = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/c_error_images/"
book_c = xlrd.open_workbook(path_c + 'c_error_array_' + filter + str(count) + '.xlsx')
sheet_c = book_c.sheet_by_name('Sheet1')
c_error_array = [sheet_c.cell_value(r, 1) for r in range(sheet_c.nrows)]
# print(er_error_array)
c_error_array = [x for x in c_error_array  if np.abs(x)<10]  


plt.figure(1, figsize=(15, 5))

plt.subplot(511)
plt.ylim(-10,10)
plt.scatter(np.arange(0,np.size(er_error_array,0),1),er_error_array, color='blue', marker='o', label='data')

plt.subplot(512)
plt.ylim(-10,10)
plt.scatter(np.arange(0,np.size(el_error_array,0),1),el_error_array, color='blue', marker='o', label='data')

plt.subplot(513)
plt.ylim(-10,10)
plt.scatter(np.arange(0,np.size(l_error_array,0),1),l_error_array, color='blue', marker='o', label='data')

plt.subplot(514)
plt.ylim(-10,10)
plt.scatter(np.arange(0,np.size(r_error_array,0),1),r_error_array, color='blue', marker='o', label='data')

plt.subplot(515)
plt.ylim(-10,10)
plt.scatter(np.arange(0,np.size(c_error_array,0),1),c_error_array, color='blue', marker='o', label='data')


plt.savefig(path_er + 'all_roi_scatter' + filter + str(count) +".png")

plt.figure(2, figsize=(15, 5)) 

plt.subplot(511)
plt.hist(er_error_array, color = 'blue', edgecolor = 'black',
         bins = int(180/5))

plt.subplot(212)
plt.hist(el_error_array, color = 'blue', edgecolor = 'black',
         bins = int(180/5))

plt.subplot(212)
plt.hist(l_error_array, color = 'blue', edgecolor = 'black',
         bins = int(180/5))

plt.subplot(212)
plt.hist(r_error_array, color = 'blue', edgecolor = 'black',
         bins = int(180/5))

plt.subplot(212)
plt.hist(c_error_array, color = 'blue', edgecolor = 'black',
         bins = int(180/5))

plt.savefig(path_er + 'all_roi_hist' + filter + str(count) +".png")

plt.show()

           