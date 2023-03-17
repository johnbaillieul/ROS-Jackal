import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

#Dataset used:
path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/"
path_tau = path + "test_results_tau/tau_value"  
path_image = path + "test_results_img/training_images_327_v_1/"

first_image_no = 48803
last_image_no = 49071

range_frames = (first_image_no, last_image_no)

#Loading data from OF
path_of = path +'/of_results/'

tau_le_of = np.load(os.path.join(path_of, 'OF_tau_le_hg.npy'))
tau_l_of = np.load(os.path.join(path_of, 'OF_tau_l_hg.npy'))
tau_c_of = np.load(os.path.join(path_of, 'OF_tau_c_hg.npy'))
tau_r_of = np.load(os.path.join(path_of, 'OF_tau_r_hg.npy'))
tau_re_of = np.load(os.path.join(path_of, 'OF_tau_re_hg.npy'))

path_shallow = path +'/shallow_results/'
tau_le_shallow = np.load(os.path.join(path_shallow, 'shallow_tau_le_HG.npy'))
tau_l_shallow = np.load(os.path.join(path_shallow, 'shallow_tau_l_HG.npy'))
tau_c_shallow = np.load(os.path.join(path_shallow, 'shallow_tau_c_HG.npy'))
tau_r_shallow = np.load(os.path.join(path_shallow, 'shallow_tau_r_HG.npy'))
tau_re_shallow = np.load(os.path.join(path_shallow, 'shallow_tau_re_HG.npy'))

#Loading data from DNN
path_dnn = path +'/dnn_results_new/'
tau_le_dnn = np.load(os.path.join(path_dnn, 'DNN_tau_le_HG.npy'))
tau_l_dnn = np.load(os.path.join(path_dnn, 'DNN_tau_l_HG.npy'))
tau_c_dnn = np.load(os.path.join(path_dnn, 'DNN_tau_c_HG.npy'))
tau_r_dnn = np.load(os.path.join(path_dnn, 'DNN_tau_r_HG.npy'))
tau_re_dnn = np.load(os.path.join(path_dnn, 'DNN_tau_re_HG.npy'))

#Loading Ground Truth
res = []
for label_no in range(range_frames[0], range_frames[1]):
    path_label = path_tau + str(label_no+1) + '.xlsx'
    df = pd.read_excel(path_label, sheet_name='Sheet1', header=None, nrows=1)
    res.append(np.array(df.values[0]))
tau_le_gt = [tmp[0] for tmp in res]
tau_l_gt = [tmp[1] for tmp in res]
tau_c_gt = [tmp[2] for tmp in res]
tau_r_gt = [tmp[3] for tmp in res]
tau_re_gt = [tmp[4] for tmp in res]

tau_le = [tau_le_of, tau_le_dnn, tau_le_gt, tau_le_shallow ]
tau_l= [tau_l_of, tau_l_dnn, tau_l_gt, tau_l_shallow]
tau_c = [tau_c_of, tau_c_dnn, tau_c_gt, tau_c_shallow]
tau_r = [tau_r_of, tau_r_dnn, tau_r_gt, tau_r_shallow]
tau_re = [tau_re_of, tau_re_dnn, tau_re_gt,tau_re_shallow]
taus = [tau_le, tau_l, tau_c, tau_r, tau_re]

tau_le = [tau_le_of, tau_le_dnn, tau_le_gt, tau_le_shallow ]
tau_l= [tau_l_of, tau_l_dnn, tau_l_gt, tau_l_shallow]
tau_c = [tau_c_of, tau_c_dnn, tau_c_gt, tau_c_shallow]
tau_r = [tau_r_of, tau_r_dnn, tau_r_gt, tau_r_shallow]
tau_re = [tau_re_of, tau_re_dnn, tau_re_gt,tau_re_shallow]
taus = [tau_le, tau_l, tau_c, tau_r, tau_re]

def plot_(y2,y3,y4,name):
    fig, ax = plt.subplots()
    print('plot_function')
    # ax.plot(y1, color='b', label='Prediction - OF', alpha=0.5)
    ax.plot(y2, color='r', label='Prediction - ResNet')
    ax.plot(y3, color='g', label='Ground truth')
    ax.plot(y4, color='b', alpha = 0.5, label='Prediction - Shallow_CNN')
    # ax.set_ylim([-1.5,15])
    # ax.set_xlabel("Frames", fontsize=20)
    # ax.set_ylabel("Tau values", fontsize=20)
    with PdfPages(name) as export_pdf:
        export_pdf.savefig()

def plot_e(y2,y3,y4,name):
    fig, ax = plt.subplots()
    print('plot_function')
    # ax.plot(y1, color='b', label='Prediction - OF', alpha=0.5)
    ax.plot(y2, color='r', label='Prediction - ResNet')
    ax.plot(y3, color='g', label='Ground truth')
    ax.plot(y4, color='b', alpha = 0.5, label='Prediction - Shallow_CNN')
    # ax.set_ylim([-1.5,5])
   
    # ax.set_xlabel("Frames", fontsize=20)
    # ax.set_ylabel("Tau values", fontsize=20)
    with PdfPages(name) as export_pdf:
        export_pdf.savefig()

# plot_e(tau_le_of, tau_le_dnn, tau_le_gt, tau_le_shallow, 'el_CP__.pdf')
# plot_e(tau_re_of, tau_re_dnn, tau_re_gt, tau_re_shallow, 'er_CP__.pdf')
# plot_(tau_l_of, tau_l_dnn, tau_l_gt,tau_l_shallow, 'l_CP__.pdf')
# plot_(tau_r_of, tau_r_dnn, tau_r_gt, tau_r_shallow, 'r_CP__.pdf')
# plot_(tau_c_of, tau_c_dnn, tau_c_gt,tau_c_shallow, 'c_CP__.pdf')
# plt.show()

plot_e(tau_le_dnn, tau_le_gt, tau_le_shallow, 'el_CP_HG_noof.pdf')
plot_e( tau_re_dnn, tau_re_gt, tau_re_shallow, 'er_CP_HG_noof.pdf')
plot_(tau_l_dnn, tau_l_gt,tau_l_shallow, 'l_CP_HG_noof.pdf')
plot_(tau_r_dnn, tau_r_gt, tau_r_shallow, 'r_CP_HG_noof.pdf')
plot_(tau_c_dnn, tau_c_gt,tau_c_shallow, 'c_CP_HG_noof.pdf')
plt.show()

def rmse(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))

    return loss

print('rmse el dnn', rmse(tau_le_gt,tau_le_dnn ) )
print('rmse er dnn', rmse(tau_re_gt,tau_re_dnn))
print('rmse l dnn', rmse(tau_l_gt,tau_l_dnn))
print('rmse r dnn', rmse(tau_r_gt,tau_r_dnn))
print('rmse c dnn', rmse(tau_c_gt,tau_c_dnn))

# print('rmse el of', rmse(tau_le_gt,tau_le_of ) )
# print('rmse er of', rmse(tau_re_gt,tau_re_of))
# print('rmse l of', rmse(tau_l_gt,tau_l_of))
# print('rmse r of', rmse(tau_r_gt,tau_r_of))
# print('rmse c of', rmse(tau_c_gt,tau_c_of))

print('rmse el shallow', rmse(tau_le_gt,tau_le_shallow ) )
print('rmse er shallow', rmse(tau_re_gt,tau_re_shallow))
print('rmse l shallow', rmse(tau_l_gt,tau_l_shallow))
print('rmse r shallow', rmse(tau_r_gt,tau_r_shallow))
print('rmse c shallow', rmse(tau_c_gt,tau_c_shallow))