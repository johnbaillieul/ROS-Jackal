#!/usr/bin/env python3
import rospy
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# %matplotlib inline 


path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/mad_ranges/"
folder_name =  'mad_range_' 
   

# print(er_error_array_1[1:,1].shape)

er_error_array = []
r_error_array = []
c_error_array = []
l_error_array = []
el_error_array = []


fig=plt.figure(figsize=(10,7))
columns = 3
rows = 2
limit = 1000000
array_str = ['er_error_array','el_error_array','r_error_array','l_error_array','c_error_array']
for j in range(1,5):
    er_error_array_1 = pd.read_excel(path_folder + folder_name + str(j) +'_1/er_error_array_mad_range_'+ str(j) + '_.xlsx')
    er_error_array_1 = er_error_array_1.to_numpy()
    er_error_array_1 = [abs(x) for x in er_error_array_1[1:,1]  if np.abs(x) < limit] 
    er_error_array.append(er_error_array_1)

    r_error_array_1 = pd.read_excel(path_folder + folder_name + str(j) +'_1/r_error_array_mad_range_'+ str(j) + '_.xlsx')
    r_error_array_1 = r_error_array_1.to_numpy()
    r_error_array_1 = [abs(x) for x in r_error_array_1[1:,1]  if np.abs(x) < limit] 
    r_error_array.append(r_error_array_1)

    c_error_array_1 = pd.read_excel(path_folder + folder_name + str(j) +'_1/c_error_array_mad_range_'+ str(j) + '_.xlsx')
    c_error_array_1 = c_error_array_1.to_numpy()
    c_error_array_1 = [abs(x) for x in c_error_array_1[1:,1]  if np.abs(x) < limit] 
    c_error_array.append(c_error_array_1)
    
    l_error_array_1 = pd.read_excel(path_folder + folder_name + str(j) +'_1/l_error_array_mad_range_'+ str(j) + '_.xlsx')
    l_error_array_1 = l_error_array_1.to_numpy()
    l_error_array_1 = [abs(x) for x in l_error_array_1[1:,1]  if np.abs(x) < limit] 
    l_error_array.append(l_error_array_1)

    el_error_array_1 = pd.read_excel(path_folder + folder_name + str(j) +'_1/el_error_array_mad_range_'+ str(j) + '_.xlsx')
    el_error_array_1 = el_error_array_1.to_numpy()
    el_error_array_1 = [abs(x) for x in el_error_array_1[1:,1]  if np.abs(x) < limit]  
    el_error_array.append(el_error_array_1)

# print('el_error_array',el_error_array)
# print('er_error_array',er_error_array)
# print('l_error_array',l_error_array)
# print('r_error_array',r_error_array)
# print('c_error_array',c_error_array)
array = [er_error_array, el_error_array, r_error_array, l_error_array, c_error_array]

for i in range(1, 6):
    fig.add_subplot(rows, columns, i)

    data_1 = array[i-1][0]
    # getting data of the histogram 
    count_1, bins_count_1 = np.histogram(data_1, bins=1000) 
    # finding the PDF of the histogram using count values 
    pdf_1 = count_1 / sum(count_1) 
    # using numpy np.cumsum to calculate the CDF 
    # We can also find using the PDF values by looping and adding 
    cdf_1 = np.cumsum(pdf_1) 
    plt.plot(bins_count_1[1:], cdf_1, label="CDF_range_1") 

    data_2 = array[i-1][1]
    # getting data of the histogram 
    count_2, bins_count_2 = np.histogram(data_2, bins=1000) 
    # finding the PDF of the histogram using count values 
    pdf_2 = count_2 / sum(count_2) 
    # using numpy np.cumsum to calculate the CDF 
    # We can also find using the PDF values by looping and adding 
    cdf_2 = np.cumsum(pdf_2) 
    plt.plot(bins_count_2[1:], cdf_2, label="CDF_range_2") 

    data_3 = array[i-1][2]
    # getting data of the histogram 
    count_3, bins_count_3 = np.histogram(data_3, bins=1000) 
    # finding the PDF of the histogram using count values 
    pdf_3 = count_3 / sum(count_3) 
    # using numpy np.cumsum to calculate the CDF 
    # We can also find using the PDF values by looping and adding 
    cdf_3 = np.cumsum(pdf_3) 
    plt.plot(bins_count_3[1:], cdf_3, label="CDF_range_3") 

    data_4 = array[i-1][3]
    # getting data of the histogram 
    count_4, bins_count_4 = np.histogram(data_1, bins=1000) 
    # finding the PDF of the histogram using count values 
    pdf_4 = count_4 / sum(count_4) 
    # using numpy np.cumsum to calculate the CDF 
    # We can also find using the PDF values by looping and adding 
    cdf_4 = np.cumsum(pdf_4) 
    plt.plot(bins_count_4[1:], cdf_4, label="CDF_range_4") 

    plt.legend()
    plt.xlim([0,10])
    plt.xlabel(array_str[i-1]) 
    plt.ylabel('steps') 
    # plt.title(array_str[i-1] + 'with diff range vlaues')

plt.savefig(path_folder + "yy.png")
plt.show()






    
