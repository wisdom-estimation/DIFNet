import torch
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1E4
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from pylab import mpl
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cpu0 = torch.device("cpu")
   print("Running on the CPU")

# Legend
Klegend = ["KNet - Train", "KNet - Validation", "KNet - Test", "Kalman Filter",'Kalman Filter partial)']
RTSlegend = ["RTSNet - Train", "RTSNet - Validation", "RTSNet - Test", "RTS Smoother","Kalman Filter"]
ERTSlegend = ["RTSNet - Train","RTSNet - Validation", "RTSNet - Test", "RTS","EKF"]
# Color
KColor = ['-ro', 'k-', 'b-','g-','c-','m-','y-']
RTSColor = ['red','darkorange','g-', 'b-']
Mycolor=['red','saddlebrown','orange','yellowgreen','mediumaquamarine','cyan','blueviolet','magenta','deeppink']
Mycolor1=['red','k','#ff9d4d','yellowgreen','blueviolet','steelblue','cyan','magenta','deeppink'] # #5b8ff9 #9270CA
Mylinestyle=['-',':','--','-.']
Mymarker=[".", ",", "o", "x", "D", "s", "*","p"]


class Plot:
    
    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    def NNPlot_epochs(self, N_Epochs_plt, MSE_KF_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=Klegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)


    def NNPlot_Hist(self, MSE_KF_data_linear_arr, MSE_KN_linear_arr):

        fileName = self.folderName + 'plt_hist_dB'

        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        sns.distplot(10 * torch.log10(MSE_KN_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = self.modelName)
        #sns.distplot(10 * torch.log10(MSE_KF_design_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter - design')
        sns.distplot(10 * torch.log10(MSE_KF_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'Kalman Filter')

        plt.title("Histogram [dB]",fontsize=32)
        plt.legend(fontsize=32)
        plt.savefig(fileName)

    def KFPlot(res_grid):

        plt.figure(figsize = (50, 20))
        x_plt = [-6, 0, 6]

        plt.plot(x_plt, res_grid[0][:], 'xg', label='minus')
        plt.plot(x_plt, res_grid[1][:], 'ob', label='base')
        plt.plot(x_plt, res_grid[2][:], '+r', label='plus')
        plt.plot(x_plt, res_grid[3][:], 'oy', label='base NN')

        plt.legend()
        plt.xlabel('Noise', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('Change', fontsize=16)
        plt.savefig('plt_grid_dB')

        print("\ndistribution 1")
        print("Kalman Filter")
        print(res_grid[0][0], "[dB]", res_grid[1][0], "[dB]", res_grid[2][0], "[dB]")
        print(res_grid[1][0] - res_grid[0][0], "[dB]", res_grid[2][0] - res_grid[1][0], "[dB]")
        print("KalmanNet", res_grid[3][0], "[dB]", "KalmanNet Diff", res_grid[3][0] - res_grid[1][0], "[dB]")

        print("\ndistribution 2")
        print("Kalman Filter")
        print(res_grid[0][1], "[dB]", res_grid[1][1], "[dB]", res_grid[2][1], "[dB]")
        print(res_grid[1][1] - res_grid[0][1], "[dB]", res_grid[2][1] - res_grid[1][1], "[dB]")
        print("KalmanNet", res_grid[3][1], "[dB]", "KalmanNet Diff", res_grid[3][1] - res_grid[1][1], "[dB]")

        print("\ndistribution 3")
        print("Kalman Filter")
        print(res_grid[0][2], "[dB]", res_grid[1][2], "[dB]", res_grid[2][2], "[dB]")
        print(res_grid[1][2] - res_grid[0][2], "[dB]", res_grid[2][2] - res_grid[1][2], "[dB]")
        print("KalmanNet", res_grid[3][2], "[dB]", "KalmanNet Diff", res_grid[3][2] - res_grid[1][2], "[dB]")

    def NNPlot_test(MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,
               MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg):


        N_Epochs_plt = 200

        ###############################
        ### Plot per epoch [linear] ###
        ###############################
        # plt.figure(figsize = (50, 20))
        plt.figure()
        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_linear_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[0], label=Klegend[2])

        # KF_ parallel
        y_plt4 = MSE_KF_linear_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[2], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [linear]', fontsize=16)
        plt.title('MSE Loss [linear] - per Epoch', fontsize=16)
        plt.savefig(fname='plt_model_test_linear.svg', dpi=600,bbox_inches='tight')

        ###########################
        ### Plot per epoch [dB] ###
        ###########################

        # plt.figure(figsize = (50, 20))
        plt.figure()
        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        # y_plt3 = KNet_MSE_test_linear_arr
        # plt.plot(x_plt, y_plt3, KColor[1], label=Klegend[2])

        # KF_parallel
        y_plt4 = MSE_KF_linear_arr
        plt.plot(x_plt, y_plt4, KColor[2], label=Klegend[3])

        # KF_
        # y_plt5 = MSE_KF_linear_arr_partialh
        # plt.plot(x_plt, y_plt5, KColor[3], label=Klegend[4])

        plt.legend()
        plt.xlabel('test_target(200)', fontsize=16)
        plt.ylabel('MSE Loss Value [linear]', fontsize=16)
        plt.title('MSE Loss [linear] - partial(H_rotated) ', fontsize=16)
        plt.savefig(fname='plt_model_test_linear.svg', dpi=600, bbox_inches='tight')


        # plt.figure(figsize = (50, 20))

        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('MSE Loss [dB] - per Epoch', fontsize=16)
        plt.savefig(fname='plt_model_test_dB.svg', dpi=600,bbox_inches='tight')

        ########################
        ### Linear Histogram ###
        ########################
        # plt.figure(figsize=(50, 20))
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [Linear]")
        plt.savefig(fname='plt_hist_linear.svg', dpi=600,bbox_inches='tight')

        fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label='KalmanNet', ax=axes[0])
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='b', label='Kalman Filter', ax=axes[1])
        plt.title("Histogram [Linear]")
        plt.savefig(fname='plt_hist_linear_1.svg', dpi=600,bbox_inches='tight')

        ####################
        ### dB Histogram ###
        ####################

        # plt.figure(figsize=(50, 20))
        sns.distplot(10 * torch.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [dB]")
        plt.savefig(fname='plt_hist_dB.svg', dpi=600,bbox_inches='tight')


        fig, axes = plt.subplots(2, 1,  sharey=True, dpi=100)
        sns.distplot(10 * torch.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet', ax=axes[0])
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter', ax=axes[1])
        plt.title("Histogram [dB]")
        plt.savefig(fname='plt_hist_dB_1.svg', dpi=600,bbox_inches='tight')

        print('End')


        # KF_design_MSE_mean_dB = 10 * torch.log10(torch.mean(MSE_KF_design_linear_arr))
        # KF_design_MSE_median_dB = 10 * torch.log10(torch.median(MSE_KF_design_linear_arr))
        # KF_design_MSE_std_dB = 10 * torch.log10(torch.std(MSE_KF_design_linear_arr))
        # print("kalman Filter - Design:",
        #       "MSE - mean", KF_design_MSE_mean_dB, "[dB]",
        #       "MSE - median", KF_design_MSE_median_dB, "[dB]",
        #       "MSE - std", KF_design_MSE_std_dB, "[dB]")
        
        # KF_data_MSE_mean_dB = 10 * torch.log10(torch.mean(MSE_KF_data_linear_arr))
        # KF_data_MSE_median_dB = 10 * torch.log10(torch.median(MSE_KF_data_linear_arr))
        # KF_data_MSE_std_dB = 10 * torch.log10(torch.std(MSE_KF_data_linear_arr))
        # print("kalman Filter - Data:",
        #       "MSE - mean", KF_data_MSE_mean_dB, "[dB]",
        #       "MSE - median", KF_data_MSE_median_dB, "[dB]",
        #       "MSE - std", KF_data_MSE_std_dB, "[dB]")
        
        # KN_MSE_mean_dB = 10 * torch.log10(torch.mean(MSE_KN_linear_arr))
        # KN_MSE_median_dB = 10 * torch.log10(torch.median(MSE_KN_linear_arr))
        # KN_MSE_std_dB = 10 * torch.log10(torch.std(MSE_KN_linear_arr))
        
        # print("kalman Net:",
        #       "MSE - mean", KN_MSE_mean_dB, "[dB]",
        #       "MSE - median", KN_MSE_median_dB, "[dB]",
        #       "MSE - std", KN_MSE_std_dB, "[dB]")

    def PlotTest_MSE_Seq(self,KFNet_test_mse,KF_test_mse,Observation_mse):
        plt.figure(1)
        plt.plot(range(KFNet_test_mse.size()[0]),KFNet_test_mse,label='KFNet_MSE_test')
        plt.plot(range(KF_test_mse.size()[0]),KF_test_mse,label='KF_MSE_test')
        plt.plot(range(Observation_mse.size()[0]),Observation_mse,label='Noise_MSE')
        plt.legend()
        plt.show()

    def PlotTest_Trajectory(self,targets_numpy,inputs_numpy,KFNet_test,KF_test_out,state_idx,state_observation_idx):
        plt.figure(figsize=(15,10))
        plt.plot(targets_numpy[0,state_idx[0],:],targets_numpy[0,state_idx[1],:], 'b', linewidth=1,label='原始轨迹')
        plt.plot(inputs_numpy[0,state_observation_idx[0],:],inputs_numpy[0,state_observation_idx[1],:], 'c*', linewidth=1,label='观测轨迹')
        plt.plot(KFNet_test[0,state_idx[0],:],KFNet_test[0,state_idx[1],:], 'r', linewidth=1,label='KalmanNet滤波轨迹')
        plt.plot(KF_test_out[0,state_idx[0],:],KF_test_out[0,state_idx[1],:], 'g', linewidth=1,label='KF滤波轨迹')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def PlotTest_MSE_Seq_MultiSensor(self,IFNet_test_mse,IF_exact_test_mse,IF_inexact_test_mse,KNet_test_mse,KF_test_mse, CI_test_mse, Observation_mse, sensor_num):
        # fig = plt.figure(1)
        # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # for i in range(sensor_num):
        #      ax.plot(range(Observation_mse[0].size()[0]),Observation_mse[i], color=Mycolor[5+i],marker=Mymarker[5+i],markersize=8,linestyle=Mylinestyle[0],label='Sensor{} Observation'.format(str(i+1))) #  传感器{}观测
        # ax.plot(range(IF_exact_test_mse.size()[0]),IF_exact_test_mse,color=Mycolor[1],marker=Mymarker[1],markersize=8,linestyle=Mylinestyle[1], label='IF_exact') #  精确信息滤波
        # ax.plot(range(IF_inexact_test_mse.size()[0]),IF_inexact_test_mse,color=Mycolor[2],marker=Mymarker[2],markersize=8,linestyle=Mylinestyle[1],label='IF_inexact') #  非精确信息滤波
        # ax.plot(range(KF_test_mse.size()[0]),KF_test_mse,color=Mycolor[4],marker=Mymarker[4],markersize=8,linestyle=Mylinestyle[2],label='Centralized_KF')
        # ax.plot(range(KNet_test_mse.size()[0]),KNet_test_mse,color=Mycolor[3],marker=Mymarker[3],markersize=8,linestyle=Mylinestyle[2],label='Centralized_KalmanNet')
        # ax.plot(range(IFNet_test_mse.size()[0]),IFNet_test_mse, color=Mycolor[0],marker=Mymarker[0],markersize=8,linestyle=Mylinestyle[1], label='IFNet') #  信息网络
        # ax.legend(fontsize=12, loc = 'upper center', bbox_to_anchor=(0.8, 1)) #  SimSun
        # plt.ylim((0, 0.1))
        # plt.xlim((-5, 105))
        # plt.yticks(fontproperties = 'Times New Roman', size = 24)
        # plt.xticks(fontproperties = 'Times New Roman', size = 24)
        # plt.xlabel('Sampling    Epoch',fontdict={'family' : 'Times New Roman', 'size': 32}) #  时刻
        # plt.ylabel('RMSE  of  Position',fontdict={'family' : 'Times New Roman', 'size': 32}) # 位置均方根误差
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # # 正常显示负号
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.show()

        fig = plt.figure(figsize=(32,20))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        for i in range(sensor_num):
             ax.plot(range(Observation_mse[0].size()[0]), Observation_mse[i], color=Mycolor1[5],marker=Mymarker[5+i], markersize=15, linestyle=Mylinestyle[0],label='Sensor{} Observation'.format(str(i+1))) #  传感器{}观测
        ax.plot(range(Observation_mse[0].size()[0]), IF_exact_test_mse, color=Mycolor1[1], marker=Mymarker[1], markersize=15, linestyle=Mylinestyle[0],
                 label='EIF-exact')  # 精确信息滤波
        ax.plot(range(Observation_mse[0].size()[0]), IF_inexact_test_mse, color=Mycolor1[2], marker=Mymarker[2], markersize=18, markerfacecolor='none', linestyle=Mylinestyle[0],
                 label='EIF-inexact')  # 非精确信息滤波
        ax.plot(range(Observation_mse[0].size()[0]), KF_test_mse, color=Mycolor1[4], marker=Mymarker[3], markersize=12, linestyle=Mylinestyle[0],
                 label='Measurement fusion') # 非精确KF滤波
        ax.plot(range(Observation_mse[0].size()[0]), CI_test_mse, color=Mycolor1[6], marker=Mymarker[0], markersize=18,linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(range(Observation_mse[0].size()[0]), KNet_test_mse, color=Mycolor1[3], marker=Mymarker[4], markersize=15, linestyle=Mylinestyle[0],
                 label='KalmanNet') # KalmanNet
        ax.plot(range(Observation_mse[0].size()[0]), IFNet_test_mse, color=Mycolor1[0], marker=Mymarker[7], markersize=15, linestyle=Mylinestyle[0],
                 label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        plt.yticks(fontproperties = 'Times New Roman', size = 48)
        plt.xticks(fontproperties = 'Times New Roman', size = 48)
        ax.set_xlabel('Sampling      Epoch', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('RMSE', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, prop=font2)  # SimSun
        plt.savefig(fname='rmse_lorenz.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotTest_MSE_Seq_MultiSensor_linear(self,IFNet_test_mse,IF_exact_test_mse,IF_inexact_test_mse,KNet_test_mse,KF_test_mse, CI_test_mse, Observation_mse, sensor_num):
        fig = plt.figure(figsize=(32,20))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        for i in range(sensor_num):
             ax.plot(range(Observation_mse[0].size()[0]), Observation_mse[i], color=Mycolor1[5],marker=Mymarker[5+i], markersize=15, linestyle=Mylinestyle[0],label='Sensor{} Observation'.format(str(i+1))) #  传感器{}观测
        ax.plot(range(Observation_mse[0].size()[0]), IF_exact_test_mse, color=Mycolor1[1], marker=Mymarker[1], markersize=15, linestyle=Mylinestyle[0],
                 label='IF-exact')  # 精确信息滤波
        ax.plot(range(Observation_mse[0].size()[0]), IF_inexact_test_mse, color=Mycolor1[2], marker=Mymarker[2], markersize=18,markerfacecolor='none', linestyle=Mylinestyle[0],
                 label='IF-inexact')  # 非精确信息滤波
        ax.plot(range(Observation_mse[0].size()[0]), KF_test_mse, color=Mycolor1[4], marker=Mymarker[3], markersize=12, linestyle=Mylinestyle[0],
                 label='Measurement fusion') # 非精确KF滤波
        ax.plot(range(Observation_mse[0].size()[0]), CI_test_mse, color=Mycolor1[6], marker=Mymarker[0], markersize=18,linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(range(Observation_mse[0].size()[0]), KNet_test_mse, color=Mycolor1[3], marker=Mymarker[4], markersize=15, linestyle=Mylinestyle[0],
                 label='KalmanNet') # KalmanNet
        ax.plot(range(Observation_mse[0].size()[0]), IFNet_test_mse, color=Mycolor1[0], marker=Mymarker[7], markersize=15, linestyle=Mylinestyle[0],
                 label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        plt.yticks(fontproperties = 'Times New Roman', size = 48)
        plt.xticks(fontproperties = 'Times New Roman', size = 48)
        ax.set_xlabel('Sampling      Epoch', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('RMSE    of    Position', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, prop=font2)  # SimSun
        plt.savefig(fname='rmse_position_linear.pdf', dpi=1200, bbox_inches='tight')
        plt.show()


    def PlotTest_MSE_Seq_Velocity_MultiSensor(self,IFNet_test_mse,IF_exact_test_mse,IF_inexact_test_mse,KNet_test_mse,KF_test_mse,CI_test_mse):
        fig = plt.figure(figsize=(32,20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(range(IF_exact_test_mse.size()[0]),IF_exact_test_mse,color=Mycolor1[1],marker=Mymarker[1],linestyle=Mylinestyle[0],markersize=15, label='IF-exact')
        ax.plot(range(IF_inexact_test_mse.size()[0]),IF_inexact_test_mse,color=Mycolor1[2],marker=Mymarker[2],linestyle=Mylinestyle[0],markerfacecolor='none',markersize=18,label='IF-inexact')
        ax.plot(range(KF_test_mse.size()[0]),KF_test_mse,color=Mycolor1[4],marker=Mymarker[3],linestyle=Mylinestyle[0],markersize=12,label='Measurement fusion')
        ax.plot(range(CI_test_mse.size()[0]), CI_test_mse, color=Mycolor1[6], marker=Mymarker[0],linestyle=Mylinestyle[0], markersize=18, label='CI')
        ax.plot(range(KNet_test_mse.size()[0]),KNet_test_mse,color=Mycolor1[3],marker=Mymarker[4],linestyle=Mylinestyle[0],markersize=15,label='KalmanNet')
        ax.plot(range(IFNet_test_mse.size()[0]),IFNet_test_mse, color=Mycolor1[0],marker=Mymarker[7],linestyle=Mylinestyle[0],markersize=15, label='IFNet')
        ax.set_xlim((-5, 105))
        # plt.legend(prop={'family' : 'Times New Roman', 'size': 24})
        left,bottom,width,height = (39.5,3.7,20.5,16) # 矩形框位置
        rect = mpatches.Rectangle((left, bottom), width, height,
                              fill=False, color="k", linewidth=2)
        plt.gca().add_patch(rect)
        ax.annotate("",
                    xy=(63, 51), xycoords='data',
                    xytext=(50, 19.5), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3",linewidth=2)
                    ) # 箭头

        plt.yticks(fontproperties = 'Times New Roman', size = 48)
        plt.xticks(fontproperties = 'Times New Roman', size = 48)
        plt.xlabel('Sampling      Epoch',fontdict={'family' : 'Times New Roman', 'size': 48}) #时刻
        plt.ylabel('RMSE    of    Velocity',fontdict={'family' : 'Times New Roman', 'size': 48}) #速度均方根误差
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        ## 图中图
        left = 40
        right = 60
        ax_new = plt.axes([0.4, 0.4, 0.4, 0.4])  #fig.add_axes([0.4, 0.4, 0.3, 0.3])
        ax_new.plot(range(left,right), IF_exact_test_mse[40:60], color=Mycolor1[1], marker=Mymarker[1], linestyle=Mylinestyle[0], markersize=15)
        ax_new.plot(range(left,right),IF_inexact_test_mse[40:60],color=Mycolor1[2],marker=Mymarker[2],linestyle=Mylinestyle[0],markerfacecolor='none',markersize=18)
        ax_new.plot(range(left,right),KF_test_mse[40:60],color=Mycolor1[4],marker=Mymarker[3],linestyle=Mylinestyle[0],markersize=12)
        ax_new.plot(range(left,right),CI_test_mse[40:60],color=Mycolor1[6],marker=Mymarker[0],linestyle=Mylinestyle[0],markersize=18)
        ax_new.plot(range(left,right),KNet_test_mse[40:60],color=Mycolor1[3],marker=Mymarker[4],linestyle=Mylinestyle[0],markersize=15)
        ax_new.plot(range(left,right),IFNet_test_mse[40:60], color=Mycolor1[0], marker=Mymarker[7], linestyle=Mylinestyle[0], markersize=15)
        plt.setp(ax_new.get_yticklabels(), fontsize=36)
        plt.setp(ax_new.get_xticklabels(), fontsize=36)
        plt.savefig(fname='rmse_vel_linear.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotTest_Trajectory_MultiSensor(self,targets_numpy,inputs_numpy,IFNet_test,IF_test_out,IF_inexact_test_out,state_idx,state_observation_idx,sensor_num):
        mark=['c*', 'r*', 'p*']
        plt.figure(figsize=(15,10))
        plt.plot(targets_numpy[0,state_idx[0],:],targets_numpy[0,state_idx[1],:], 'b', linewidth=1,label='Ground truth')
        for i in range(sensor_num):
            plt.plot(inputs_numpy[i,0,state_observation_idx[0],:],inputs_numpy[i, 0,state_observation_idx[1],:], mark[i], linewidth=1,label='Sensor'+str(i))
        plt.plot(IFNet_test[0,state_idx[0],:],IFNet_test[0,state_idx[1],:], 'r', linewidth=1,label='IFNet')
        plt.plot(IF_test_out[0,state_idx[0],:],IF_test_out[0,state_idx[1],:], 'm', linewidth=1,label='IF_exact')
        plt.plot(IF_inexact_test_out[0,state_idx[0],:],IF_inexact_test_out[0,state_idx[1],:], 'g', linewidth=1,label='IF_inexact')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def PlotTest_Trajectory_Multisensor_Pendulum(self,targets_numpy,IFNet_test_out,IF_test_out,IF_inexact_test_out,KNet_test_out,state_idx,state_observation_idx):
        fig = plt.figure(figsize=(32,10))
        ax = fig.add_axes([0.1,0.55,0.8,0.35])
        l1=ax.plot(range(targets_numpy[0,:,:].shape[1]),targets_numpy[0,state_idx[0],:], 'b--', linewidth=2, label='angle ground truth')[0]
        ax.plot(range(IF_test_out[0,:,:].shape[1]),IF_test_out[0,state_observation_idx[0],:], color='limegreen', linewidth=2,label='EIF_exact')
        ax.plot(range(IF_inexact_test_out[0,:,:].shape[1]),IF_inexact_test_out[0,state_observation_idx[0],:], 'm', linewidth=2,label='EIF_inexact')
        ax.plot(range(KNet_test_out[0,:,:].shape[1]),KNet_test_out[0,state_observation_idx[0],:], 'c', linewidth=2,label='KalmanNet')
        ax.plot(range(IFNet_test_out[0,:,:].shape[1]),IFNet_test_out[0,state_observation_idx[0],:], 'r', linewidth=2,label='IFNet')
        plt.xlim((0, 100))
        plt.ylim((-0.5,0.5))
        plt.xlabel('t (s)',fontdict={'family': 'Times New Roman', 'size': 32})
        plt.ylabel('angle (rad)',fontdict={'family': 'Times New Roman', 'size': 32})
        # plt.ticklabel_format(style='sci', axis='both',scilimits=(0,0),useMathText=True)
        # plt.legend(prop={'family': 'Times New Roman', 'size': 24}, ncol=2)  # SimSun
        plt.yticks(fontproperties='Times New Roman', size=24)
        plt.xticks(fontproperties='Times New Roman', size=24)
        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        ax_new = fig.add_axes([0.1,0.1,0.8,0.4])
        # ax = fig.add_axes([0.1,0.1,0.8,0.8])
        l2 = ax_new.plot(range(targets_numpy[0,:,:].shape[1]),targets_numpy[0,state_idx[1],:], 'p--', linewidth=2, label='angular velocity ground truth')[0]
        l3 = ax_new.plot(range(IF_test_out[0,:,:].shape[1]),IF_test_out[0,state_observation_idx[1],:], color='limegreen', linewidth=2,label='IF_exact')[0]
        l4 = ax_new.plot(range(IF_inexact_test_out[0,:,:].shape[1]),IF_inexact_test_out[0,state_observation_idx[1],:], 'm', linewidth=2,label='IF_inexact')[0]
        l5 = ax_new.plot(range(KNet_test_out[0,:,:].shape[1]),KNet_test_out[0,state_observation_idx[1],:], 'c', linewidth=2,label='KalmanNet')[0]
        l6 = ax_new.plot(range(IFNet_test_out[0,:,:].shape[1]),IFNet_test_out[0,state_observation_idx[1],:], 'r', linewidth=2,label='IFNet')[0]
        plt.xlim((0, 100))
        # plt.ylim((-0.5,0.5))
        plt.xlabel('t (s)',fontdict={'family': 'Times New Roman', 'size': 32})
        plt.ylabel('angular velocity (rad/s)',fontdict={'family': 'Times New Roman', 'size': 32})
        # plt.ticklabel_format(style='sci', axis='both',scilimits=(0,0),useMathText=True)
        # plt.title('True Trajectory', pad=10, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 15,
                 }
        lines = [l1,l2,l3,l4,l5,l6]
        labels = ['angle ground truth', 'angular velocity ground truth','IF_exact','IF_inexact','KalmanNet','IFNet']
        # axLine1, axLabel1 = fig.axes[0].get_legend_handles_labels()
        # axLine2, axLabel2 = fig.axes[-1].get_legend_handles_labels()
        # lines.extend(axLine1[0])
        # labels.extend(axLabel1[0])
        # lines.extend(axLine2)
        # labels.extend(axLabel2)
        fig.legend(lines, labels, loc='upper center',
                    bbox_to_anchor=(0.5, 1.0), ncol=6, fancybox=True, prop=font2)
        # ax_new.legend(loc='upper center',
        #             bbox_to_anchor=(0.5, 1.07), ncol=6, framealpha=1, fancybox=True, prop=font2)
        plt.yticks(fontproperties='Times New Roman', size=24)
        plt.xticks(fontproperties='Times New Roman', size=24)
        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False
        plt.savefig('traj.eps', dpi=600, bbox_inches='tight')
        plt.show()

    def PlotTest_Trajectory_Multisensor_Pendulum_new(self,targets_numpy,IFNet_test_out,IF_test_out,IF_inexact_test_out,CI_test_out,KNet_test_out,state_idx,state_observation_idx):
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(range(targets_numpy[0, :, :].shape[1]), targets_numpy[0, state_idx[0], :], 'b--', linewidth=2,
                     label='angle ground truth')
        ax.plot(range(IF_test_out[0, :, :].shape[1]), IF_test_out[0, state_observation_idx[0], :], color='limegreen',
                linewidth=2, label='EIF_exact')
        ax.plot(range(IF_inexact_test_out[0, :, :].shape[1]), IF_inexact_test_out[0, state_observation_idx[0], :], 'm',
                linewidth=2, label='EIF_inexact')
        ax.plot(range(CI_test_out[0, :, :].shape[1]), CI_test_out[0, state_observation_idx[0], :], 'k',
                linewidth=2, label='CI')
        ax.plot(range(KNet_test_out[0, :, :].shape[1]), KNet_test_out[0, state_observation_idx[0], :], 'c', linewidth=2,
                label='KalmanNet')
        ax.plot(range(IFNet_test_out[0, :, :].shape[1]), IFNet_test_out[0, state_observation_idx[0], :], 'r', linewidth=2,
                label='IFNet')
        plt.xlim((0, 100))
        plt.ylim((-0.5, 0.5))
        plt.xlabel('t (s)', fontdict={'family': 'Times New Roman', 'size': 32})
        plt.ylabel('angle (rad)', fontdict={'family': 'Times New Roman', 'size': 32})
        # plt.ticklabel_format(style='sci', axis='both',scilimits=(0,0),useMathText=True)
        plt.legend(prop={'family': 'Times New Roman', 'size': 24}, ncol=3)  # SimSun
        plt.yticks(fontproperties='Times New Roman', size=24)
        plt.xticks(fontproperties='Times New Roman', size=24)
        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False
        left,bottom,width,height = (70,-0.4,14,0.16)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False, color="k", linewidth=1)
        plt.gca().add_patch(rect)
        ax.annotate("",
                    xy=(50, -0.3), xycoords='data',
                    xytext=(70, -0.3), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),
                    )
        #图中图
        left = 70
        right = 85
        ax_new = plt.axes([0.15, 0.15, 0.35, 0.35])  # fig.add_axes([0.4, 0.4, 0.3, 0.3])
        ax_new.plot(range(left, right), targets_numpy[0, state_idx[0], 70:85], 'b--', linewidth=2,
                label='angle ground truth')
        ax_new.plot(range(left, right), IF_test_out[0, state_observation_idx[0], 70:85], color='limegreen',
                linewidth=2, label='EIF_exact')
        ax_new.plot(range(left, right), IF_inexact_test_out[0, state_observation_idx[0], 70:85], 'm',
                linewidth=2, label='EIF_inexact')
        ax_new.plot(range(left, right), CI_test_out[0, state_observation_idx[0], 70:85], 'k',
                linewidth=2, label='CI')
        ax_new.plot(range(left, right), KNet_test_out[0, state_observation_idx[0], 70:85], 'c', linewidth=2,
                label='KalmanNet')
        ax_new.plot(range(left, right), IFNet_test_out[0, state_observation_idx[0], 70:85], 'r', linewidth=2,
                label='IFNet')
        plt.savefig(fname='traj_angle.eps', dpi=600, bbox_inches='tight')
        plt.show()
        ##########################################################################################################
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(range(targets_numpy[0, :, :].shape[1]), targets_numpy[0, state_idx[1], :], 'b--', linewidth=2,
                label='angular velocity ground truth')
        ax.plot(range(IF_test_out[0, :, :].shape[1]), IF_test_out[0, state_observation_idx[1], :], color='limegreen',
                linewidth=2, label='EIF_exact')
        ax.plot(range(IF_inexact_test_out[0, :, :].shape[1]), IF_inexact_test_out[0, state_observation_idx[1], :], 'm',
                linewidth=2, label='EIF_inexact')
        ax.plot(range(CI_test_out[0, :, :].shape[1]), CI_test_out[0, state_observation_idx[1], :], 'k',
                linewidth=2, label='CI')
        ax.plot(range(KNet_test_out[0, :, :].shape[1]), KNet_test_out[0, state_observation_idx[1], :], 'c', linewidth=2,
                label='KalmanNet')
        ax.plot(range(IFNet_test_out[0, :, :].shape[1]), IFNet_test_out[0, state_observation_idx[1], :], 'r', linewidth=2,
                label='IFNet')
        plt.xlim((0, 100))
        # plt.ylim((-0.5, 0.5))
        plt.xlabel('t (s)', fontdict={'family': 'Times New Roman', 'size': 32})
        plt.ylabel('angular velocity (rad/s)', fontdict={'family': 'Times New Roman', 'size': 32})
        # plt.ticklabel_format(style='sci', axis='both',scilimits=(0,0),useMathText=True)
        plt.legend(prop={'family': 'Times New Roman', 'size': 24}, ncol=3)  # SimSun
        plt.yticks(fontproperties='Times New Roman', size=24)
        plt.xticks(fontproperties='Times New Roman', size=24)
        # 正常显示负号
        plt.rcParams['axes.unicode_minus'] = False
        left, bottom, width, height = (48, -1.2, 12, 0.2)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False, color="k", linewidth=1)
        plt.gca().add_patch(rect)
        ax.annotate("",
                    xy=(54, -0.34), xycoords='data',
                    xytext=(54, -1), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3"),
                    )
        # 图中图
        left = 48
        right = 60
        ax_new = plt.axes([0.4, 0.4, 0.27, 0.27])  # fig.add_axes([0.4, 0.4, 0.3, 0.3])
        # ax_new.plot(range(left, right), targets_numpy[0, state_idx[0], 44:60], 'b--', linewidth=2,
        #             label='angle ground truth')
        ax_new.plot(range(left, right), IF_test_out[0, state_observation_idx[1], 48:60], color='limegreen',
                    linewidth=2, label='EIF_exact')
        ax_new.plot(range(left, right), IF_inexact_test_out[0, state_observation_idx[1], 48:60], 'm',
                    linewidth=2, label='EIF_inexact')
        ax_new.plot(range(left, right), CI_test_out[0, state_observation_idx[1], 48:60], 'k',
                    linewidth=2, label='CI')
        ax_new.plot(range(left, right), KNet_test_out[0, state_observation_idx[1], 48:60], 'c', linewidth=2,
                    label='KalmanNet')
        ax_new.plot(range(left, right), IFNet_test_out[0, state_observation_idx[1], 48:60], 'r', linewidth=2,
                    label='IFNet')
        plt.savefig(fname='traj_angular.eps', dpi=600, bbox_inches='tight')
        plt.show()

    def PlotFusiontime_MultiSensor(self, EIF_fusiontime, EKF_fusiontime,CI_inexact_fusiontime,KNet_fusiontime,InformationNet_fusiontime,sensor_num):
        fontSize = 48
        ##################
        ###  Histogram ###
        ##################
        method = ['EIF measurement fusion','EIF estimate fusion','CI estimate fusion','KalmanNet measurement fusion', 'IFNet estimate fusion']
        method = method * len(sensor_num)
        fusion_time = []
        sensor = []
        for i in range(len(sensor_num)):
            fusion_time.append(EKF_fusiontime[i])
            fusion_time.append(EIF_fusiontime[i])
            fusion_time.append(CI_inexact_fusiontime[i])
            fusion_time.append(KNet_fusiontime[i])
            fusion_time.append(InformationNet_fusiontime[i])
            sensor.append(sensor_num[i])
            sensor.append(sensor_num[i])
            sensor.append(sensor_num[i])
            sensor.append(sensor_num[i])
            sensor.append(sensor_num[i])

        data = pd.DataFrame({'method':method,'fusion_time':fusion_time,'sensor_num':sensor})
        # 给定的颜色列表
        custom_colors = ['blueviolet', 'black', 'cyan', 'yellowgreen', 'red']
        # 自定义 palette
        custom_palette = {method[i]: custom_colors[i] for i in range(len(custom_colors))}

        plt.figure(figsize=(32, 20))
        sns.barplot(x='sensor_num',y='fusion_time',hue='method',data=data,palette=custom_palette)
        plt.legend(fontsize=38)
        plt.xlabel('Sensor   Number', fontsize=fontSize)
        plt.ylabel('Fusion   Time (ms)', fontsize=fontSize)
        plt.tick_params(labelsize=fontSize)
        plt.savefig(fname='fusion_time_cpu.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotParameter_MultiSensor(self, KNet_parameter, InformationNet_parameter,sensor_num):
        fontSize = 48
        ##################
        ###  Histogram ###
        ##################
        method = ['KalmanNet', 'IFNet']
        method = method * len(sensor_num)
        parameter = []
        sensor = []
        for i in range(len(sensor_num)):
            parameter.append(KNet_parameter[i])
            parameter.append(InformationNet_parameter[i])
            sensor.append(sensor_num[i])
            sensor.append(sensor_num[i])

        data = pd.DataFrame({'method':method,'parameter':parameter,'sensor_num':sensor})
        # 给定的颜色列表
        custom_colors = ['yellowgreen', 'red']
        # 自定义 palette
        custom_palette = {method[i]: custom_colors[i] for i in range(len(custom_colors))}

        plt.figure(figsize=(32, 20))
        ax = sns.barplot(x='sensor_num',y='parameter',hue='method',data=data,palette=custom_palette)
        plt.legend(fontsize=48, loc="upper left")
        plt.xlabel('Sensor Number', fontsize=fontSize)
        plt.ylabel(r'Number  of  Parameter  ($log_{10}(\cdot)$)', fontsize=fontSize)
        plt.tick_params(labelsize=fontSize)
        yfmt = ScalarFormatter()
        yfmt.set_powerlimits((-3, 3))  # Or whatever your limits are . . .
        plt.gca().yaxis.set_major_formatter(yfmt)

        # 调整字体大小
        plt.tick_params(axis='both', which='major', labelsize=48)  # 主刻度标签的字体大小
        plt.tick_params(axis='both', which='minor', labelsize=48)  # 次刻度标签的字体大小
        plt.ylim((0, 10))
        plt.savefig(fname='parameter.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotMSE_Correlation(self, EIF_MSE, EKF_MSE,CI_inexact_MSE,KNet_MSE,InformationNet_MSE):
        fig = plt.figure(figsize=(32, 20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(np.array([-0.19, 0, 0.3, 0.6, 0.99]), EIF_MSE, color=Mycolor1[1], marker=Mymarker[1],
                markersize=15, linestyle=Mylinestyle[0],
                label='EIF-exact')  # 精确信息滤波
        ax.plot(np.array([-0.19, 0, 0.3, 0.6, 0.99]), EKF_MSE, color=Mycolor1[2], marker=Mymarker[2], markerfacecolor = "none",
                markersize=18, linestyle=Mylinestyle[0],
                label='EIF-inexact')  # 非精确KF滤波
        ax.plot(np.array([-0.19, 0, 0.3, 0.6, 0.99]), CI_inexact_MSE, color=Mycolor1[6], marker=Mymarker[0], markersize=18,
                linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(np.array([-0.19, 0, 0.3, 0.6, 0.99]), KNet_MSE, color=Mycolor1[3], marker=Mymarker[4], markersize=15,
                linestyle=Mylinestyle[0],
                label='KalmanNet')  # KalmanNet
        ax.plot(np.array([-0.19, 0, 0.3, 0.6, 0.99]), InformationNet_MSE, color=Mycolor1[0], marker=Mymarker[7],
                markersize=15, linestyle=Mylinestyle[0],
                label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        ax.set_xlabel(r'$\rho$', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('MSE   (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        # 设置坐标轴刻度
        my_x_ticks = np.array([-0.19, 0, 0.3, 0.6, 0.99])
        plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0,10,1)
        # plt.yticks(my_y_ticks)
        ax.legend(loc='upper center', bbox_to_anchor=(0.22, 0.2), ncol=2, fancybox=True, prop=font2)  # SimSun
        # 获取x,y轴的刻度标签
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()
        # 调整x轴刻度标签字体大小
        for xtick in xticks:
            xtick.set_fontsize(48)
        for ytick in yticks:
            ytick.set_fontsize(48)
        plt.savefig(fname='mse_correlation.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotMSE_SNR(self, EIF_MSE, EKF_MSE,CI_inexact_MSE,KNet_MSE,InformationNet_MSE):
        fig = plt.figure(figsize=(32, 20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(np.arange(-20,30,10), EIF_MSE, color=Mycolor1[1], marker=Mymarker[1],
                markersize=15, linestyle=Mylinestyle[0],
                label='EIF-exact')  # 精确信息滤波
        ax.plot(np.arange(-20,30,10), EKF_MSE, color=Mycolor1[2], marker=Mymarker[2], markerfacecolor = 'none',
                markersize=18, linestyle=Mylinestyle[0],
                label='EIF-inexact')  # 非精确KF滤波
        ax.plot(np.arange(-20,30,10), CI_inexact_MSE, color=Mycolor1[6], marker=Mymarker[0], markersize=18,
                linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(np.arange(-20,30,10), KNet_MSE, color=Mycolor1[3], marker=Mymarker[4], markersize=15,
                linestyle=Mylinestyle[0],
                label='KalmanNet')  # KalmanNet
        ax.plot(np.arange(-20,30,10), InformationNet_MSE, color=Mycolor1[0], marker=Mymarker[7],
                markersize=15, linestyle=Mylinestyle[0],
                label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        ax.set_xlabel(r'$q^2/r^2$ (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('MSE   (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        # 设置坐标轴刻度
        my_x_ticks = np.arange(-20, 30, 10)
        plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0,10,1)
        # plt.yticks(my_y_ticks)
        ax.legend(loc='upper center', bbox_to_anchor=(0.22, 1.0), ncol=2, fancybox=True, prop=font2)  # SimSun
        # 获取x,y轴的刻度标签
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()
        # 调整x轴刻度标签字体大小
        for xtick in xticks:
            xtick.set_fontsize(48)
        for ytick in yticks:
            ytick.set_fontsize(48)
        plt.savefig(fname='mse_snr.pdf', dpi=1200, bbox_inches='tight')
        plt.show()
    def PlotMSE_filter_SNR(self, EIF_MSE, EKF_MSE,CI_inexact_MSE,KNet_MSE,InformationNet_MSE):
        fig = plt.figure(figsize=(32, 20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(np.arange(-20,30,10), EIF_MSE, color=Mycolor1[1], marker=Mymarker[1],
                markersize=15, linestyle=Mylinestyle[0],
                label='EIF-exact')  # 精确信息滤波
        ax.plot(np.arange(-20,30,10), EKF_MSE, color=Mycolor1[2], marker=Mymarker[2],
                markersize=18, markerfacecolor ='none', linestyle=Mylinestyle[0],
                label='EIF-inexact')  # 非精确KF滤波
        ax.plot(np.arange(-20,30,10), CI_inexact_MSE, color=Mycolor1[6], marker=Mymarker[0], markersize=18,
                linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(np.arange(-20,30,10), KNet_MSE, color=Mycolor1[3], marker=Mymarker[4], markersize=15,
                linestyle=Mylinestyle[0],
                label='KalmanNet')  # KalmanNet
        ax.plot(np.arange(-20,30,10), InformationNet_MSE, color=Mycolor1[0], marker=Mymarker[7],
                markersize=15, linestyle=Mylinestyle[0],
                label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        ax.set_xlabel(r'$q^2/r^2$ (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('MSE   (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        # 设置坐标轴刻度
        my_x_ticks = np.arange(-20, 30, 10)
        plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0,10,1)
        # plt.yticks(my_y_ticks)
        ax.legend(loc='upper center', bbox_to_anchor=(0.22, 1.0), ncol=2, fancybox=True, prop=font2)  # SimSun
        # 获取x,y轴的刻度标签
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()
        # 调整x轴刻度标签字体大小
        for xtick in xticks:
            xtick.set_fontsize(48)
        for ytick in yticks:
            ytick.set_fontsize(48)
        plt.savefig(fname='mse_filter_snr.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotMSE_positive_changeq(self, EIF_MSE, EKF_MSE,CI_inexact_MSE,KNet_MSE,InformationNet_MSE):
        fig = plt.figure(figsize=(32, 20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(np.arange(-20,30,10), EIF_MSE, color=Mycolor1[1], marker=Mymarker[1],
                markersize=15, linestyle=Mylinestyle[0],
                label='EIF-exact')  # 精确信息滤波
        ax.plot(np.arange(-20,30,10), EKF_MSE, color=Mycolor1[2], marker=Mymarker[2],
                markersize=18, markerfacecolor ='none', linestyle=Mylinestyle[0],
                label='EIF-inexact')  # 非精确KF滤波
        ax.plot(np.arange(-20,30,10), CI_inexact_MSE, color=Mycolor1[6], marker=Mymarker[0], markersize=18,
                linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(np.arange(-20,30,10), KNet_MSE, color=Mycolor1[3], marker=Mymarker[4], markersize=15,
                linestyle=Mylinestyle[0],
                label='KalmanNet')  # KalmanNet
        ax.plot(np.arange(-20,30,10), InformationNet_MSE, color=Mycolor1[0], marker=Mymarker[7],
                markersize=15, linestyle=Mylinestyle[0],
                label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        ax.set_xlabel(r'$q^2$   (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('MSE   (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        # 设置坐标轴刻度
        my_x_ticks = np.arange(-20, 30, 10)
        plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0,10,1)
        # plt.yticks(my_y_ticks)
        ax.legend(loc='upper center', bbox_to_anchor=(0.22, 1.0), ncol=2, fancybox=True, prop=font2)  # SimSun
        # 获取x,y轴的刻度标签
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()
        # 调整x轴刻度标签字体大小
        for xtick in xticks:
            xtick.set_fontsize(48)
        for ytick in yticks:
            ytick.set_fontsize(48)
        plt.savefig(fname='mse_positve_changeq.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotMSE_positive_changer(self, EIF_MSE, EKF_MSE,CI_inexact_MSE,KNet_MSE,InformationNet_MSE):
        fig = plt.figure(figsize=(32, 20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(np.arange(-20,30,10), EIF_MSE, color=Mycolor1[1], marker=Mymarker[1],
                markersize=15, linestyle=Mylinestyle[0],
                label='EIF-exact')  # 精确信息滤波
        ax.plot(np.arange(-20,30,10), EKF_MSE, color=Mycolor1[2], marker=Mymarker[2],
                markersize=18, markerfacecolor ='none', linestyle=Mylinestyle[0],
                label='EIF-inexact')  # 非精确KF滤波
        ax.plot(np.arange(-20,30,10), CI_inexact_MSE, color=Mycolor1[6], marker=Mymarker[0], markersize=18,
                linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(np.arange(-20,30,10), KNet_MSE, color=Mycolor1[3], marker=Mymarker[4], markersize=15,
                linestyle=Mylinestyle[0],
                label='KalmanNet')  # KalmanNet
        ax.plot(np.arange(-20,30,10), InformationNet_MSE, color=Mycolor1[0], marker=Mymarker[7],
                markersize=15, linestyle=Mylinestyle[0],
                label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        ax.set_xlabel(r'$r^2$   (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('MSE   (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        # 设置坐标轴刻度
        my_x_ticks = np.arange(-20, 30, 10)
        plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0,10,1)
        # plt.yticks(my_y_ticks)
        ax.legend(loc='upper center', bbox_to_anchor=(0.22, 1.0), ncol=2, fancybox=True, prop=font2)  # SimSun
        # 获取x,y轴的刻度标签
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()
        # 调整x轴刻度标签字体大小
        for xtick in xticks:
            xtick.set_fontsize(48)
        for ytick in yticks:
            ytick.set_fontsize(48)
        plt.savefig(fname='mse_positve_changer.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

    def PlotMSE_MultiSensor(self,EIF_MSE,EKF_MSE,CI_MSE,KNet_MSE,InformationNet_MSE, sensor_num):
        fontSize = 32
        # ##################
        # ###  Histogram ###
        # ##################
        # method = ['EKF measurement fusion', 'EIF_exact estimation fusion', 'CI estimation fusion',
        #           'KalmanNet measurement fusion', 'IFNet estimation fusion']
        # method = method * len(sensor_num)
        # mse = []
        # sensor = []
        # for i in range(len(sensor_num)):
        #     mse.append(EKF_mse[i])
        #     mse.append(EIF_mse[i])
        #     mse.append(CI_mse[i])
        #     mse.append(KNet_mse[i])
        #     mse.append(InformationNet_mse[i])
        #     sensor.append(sensor_num[i])
        #     sensor.append(sensor_num[i])
        #     sensor.append(sensor_num[i])
        #     sensor.append(sensor_num[i])
        #     sensor.append(sensor_num[i])
        #
        # data = pd.DataFrame({'method': method, 'mse': mse, 'sensor_num': sensor})
        # # 给定的颜色列表
        # custom_colors = ['blueviolet', 'black', 'cyan', 'yellowgreen', 'red']
        # # 自定义 palette
        # custom_palette = {method[i]: custom_colors[i] for i in range(len(custom_colors))}
        #
        # plt.figure(figsize=(16, 10))
        # sns.barplot(x='sensor_num', y='mse', hue='method', data=data, palette=custom_palette)
        # plt.legend(fontsize=fontSize)
        # plt.xlabel('Sensor num', fontsize=fontSize)
        # plt.ylabel('MSE (dB)', fontsize=fontSize)
        # # my_y_ticks = np.arange(-10,0,2)
        # # plt.yticks(my_y_ticks)
        # plt.tick_params(labelsize=fontSize)
        # plt.savefig(fname='mse_multisensor.pdf', dpi=600, bbox_inches='tight')
        # plt.show()
        fig = plt.figure(figsize=(32, 20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(np.arange(2,10,2), EIF_MSE, color=Mycolor1[1], marker=Mymarker[1],
                markersize=15, linestyle=Mylinestyle[0],
                label='EIF-exact')  # 精确信息滤波
        ax.plot(np.arange(2,10,2), EKF_MSE, color=Mycolor1[2], marker=Mymarker[2],
                markersize=18, markerfacecolor = 'none', linestyle=Mylinestyle[0],
                label='EIF-inexact')  # 非精确KF滤波
        ax.plot(np.arange(2,10,2), CI_MSE, color=Mycolor1[6], marker=Mymarker[0], markersize=15,
                linestyle=Mylinestyle[0],
                label='CI')  # 非精确CI滤波
        ax.plot(np.arange(2,10,2), KNet_MSE, color=Mycolor1[3], marker=Mymarker[4], markersize=15,
                linestyle=Mylinestyle[0],
                label='KalmanNet')  # KalmanNet
        ax.plot(np.arange(2,10,2), InformationNet_MSE, color=Mycolor1[0], marker=Mymarker[7],
                markersize=15, linestyle=Mylinestyle[0],
                label='IFNet')  # IFNet

        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 38,
                 }
        ax.set_xlabel(r'Sensor   Number', fontdict={'family': 'Times New Roman', 'size': 48})  # 时刻
        ax.set_ylabel('MSE (dB)', fontdict={'family': 'Times New Roman', 'size': 48})  # 位置均方根误差
        # 设置坐标轴刻度
        my_x_ticks = np.arange(2,10,2)
        plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0,10,1)
        # plt.yticks(my_y_ticks)
        ax.legend(loc='upper center', bbox_to_anchor=(0.75, 0.8), ncol=2, fancybox=True, prop=font2)  # SimSun
        # 获取x,y轴的刻度标签
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()
        #调整x轴刻度标签字体大小
        for xtick in xticks:
            xtick.set_fontsize(48)
        for ytick in yticks:
            ytick.set_fontsize(48)
        plt.savefig(fname='mse_multisensor.pdf', dpi=1200, bbox_inches='tight')
        plt.show()


class Plot_RTS(Plot):

    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    def NNPlot_epochs(self, N_MiniBatchTrain_plt, BatchSize, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):
        N_Epochs_plt = np.floor(N_MiniBatchTrain_plt/BatchSize).astype(int) # number of epochs

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=RTSlegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=RTSlegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=RTSlegend[2])

        # RTS
        y_plt4 = MSE_RTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, "g", label=RTSlegend[3])

        # KF
        y_plt5 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, "orange", label=RTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)


    def NNPlot_Hist(self, MSE_KF_linear_arr, MSE_RTS_data_linear_arr, MSE_RTSNet_linear_arr):

        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        sns.distplot(10 * torch.log10(MSE_RTSNet_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 5}, color='b', label = 'RTSNet')
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'orange', label = 'Kalman Filter')
        sns.distplot(10 * torch.log10(MSE_RTS_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3.2,"linestyle":'--'}, color= 'g', label = 'RTS Smoother')

        plt.title(self.modelName + ":" +"Histogram [dB]",fontsize=fontSize)
        plt.legend(fontsize=fontSize)
        plt.xlabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.ylabel('Percentage', fontsize=fontSize)
        plt.tick_params(labelsize=fontSize)
        plt.grid(True)
        plt.savefig(fileName)

    def KF_RTS_Plot_Linear(self, r, MSE_KF_RTS_dB,PlotResultName):
        fileName = self.folderName + PlotResultName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_KF_RTS_dB[0,:], '-^',color='orange',linewidth=1, markersize=12, label=r'2x2, KF')
        plt.plot(x_plt, MSE_KF_RTS_dB[1,:], '--go',markerfacecolor='none',linewidth=3, markersize=12, label=r'2x2, RTS')
        plt.plot(x_plt, MSE_KF_RTS_dB[2,:], '-bo',linewidth=1, markersize=12, label=r'2x2, RTSNet')

        plt.legend(fontsize=32)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        # plt.title('Comparing Kalman Filter and RTS Smoother', fontsize=32)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.savefig(fileName)

    def rotate_RTS_Plot_F(self, r, MSE_RTS_dB,rotateName):
        fileName = self.folderName + rotateName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_RTS_dB[0,:], '-r^', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB[1,:], '-gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB[2,:], '-bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{F}_{\alpha=10^\circ}$)')

        plt.legend(fontsize=16)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.savefig(fileName)

    def rotate_RTS_Plot_H(self, r, MSE_RTS_dB,rotateName):
        fileName = self.folderName + rotateName
        magnifying_glass, main_H = plt.subplots(figsize = [25, 10])
        # main_H = plt.figure(figsize = [25, 10])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_H.plot(x_plt, NoiseFloor, '--r', linewidth=2, markersize=12, label=r'Noise Floor')
        main_H.plot(x_plt, MSE_RTS_dB[0,:], '-g^', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB] , 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        main_H.plot(x_plt, MSE_RTS_dB[1,:], '-yx', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        main_H.plot(x_plt, MSE_RTS_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')

        main_H.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-20, 15))
        main_H.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)

        ax2 = plt.axes([.15, .15, .27, .27])
        x1, x2, y1, y2 =  -0.2, 0.2, -5, 8
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=2, markersize=12, label=r'Noise Floor')
        ax2.plot(x_plt, MSE_RTS_dB[0,:], '-g^', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB] , 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        ax2.plot(x_plt, MSE_RTS_dB[1,:], '-yx', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        ax2.plot(x_plt, MSE_RTS_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')
        ax2.grid(True)
        plt.savefig(fileName)

    def rotate_RTS_Plot_FHCompare(self, r, MSE_RTS_dB_F,MSE_RTS_dB_H,rotateName):
        fileName = self.folderName + rotateName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r)

        plt.plot(x_plt, MSE_RTS_dB_F[0,:], '-r^', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_F[1,:], '-gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_F[2,:], '-bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[0,:], '--r^', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[1,:], '--gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[2,:], '--bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')

        plt.legend(fontsize=16)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        plt.savefig(fileName)




class Plot_extended(Plot_RTS):
    def EKFPlot_Hist(self, MSE_EKF_linear_arr):
        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        sns.distplot(10 * np.log10(MSE_EKF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Extended Kalman Filter')
        plt.title(self.modelName + ":" +"Histogram [dB]",fontsize=fontSize)
        plt.legend(fontsize=fontSize)
        plt.savefig(fileName)

    def KF_RTS_Plot(self, r, MSE_KF_RTS_dB):
        fileName = self.folderName + 'Nonlinear_KF_RTS_Compare_dB'
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_KF_RTS_dB[0,:], '-gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], Toy Model, EKF')
        plt.plot(x_plt, MSE_KF_RTS_dB[1,:], '--bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], Toy Model, Extended RTS')

        plt.legend(fontsize=32)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{q^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('Comparing Extended Kalman Filter and Extended RTS Smoother', fontsize=32)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.savefig(fileName)

    def NNPlot_trainsteps(self, N_MiniBatchTrain_plt, MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):
        N_Epochs_plt = N_MiniBatchTrain_plt

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=ERTSlegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=ERTSlegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=ERTSlegend[2])

        # RTS
        y_plt4 = MSE_ERTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=ERTSlegend[3])

        # EKF
        y_plt5 = MSE_EKF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, KColor[4], label=ERTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Steps', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.grid(True)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Step", fontsize=fontSize)
        plt.savefig(fileName)



    def NNPlot_epochs(self, N_E,N_MiniBatchTrain_plt, BatchSize, MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):
        N_Epochs_plt = np.floor(N_MiniBatchTrain_plt*BatchSize/N_E).astype(int) # number of epochs
        print(N_Epochs_plt)
        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[np.linspace(0,N_MiniBatchTrain_plt-1,N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=ERTSlegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[np.linspace(0,N_MiniBatchTrain_plt-1,N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=ERTSlegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=ERTSlegend[2])

        # RTS
        y_plt4 = MSE_ERTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=ERTSlegend[3])

        # EKF
        y_plt5 = MSE_EKF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, KColor[4], label=ERTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.grid(True)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)

    def NNPlot_Hist(self, MSE_EKF_linear_arr, MSE_ERTS_data_linear_arr, MSE_RTSNet_linear_arr):

        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        sns.distplot(10 * torch.log10(MSE_RTSNet_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 5}, color='b', label = self.modelName)
        sns.distplot(10 * torch.log10(MSE_EKF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'orange', label = 'EKF')
        sns.distplot(10 * torch.log10(MSE_ERTS_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3.2,"linestyle":'--'},color= 'g', label = 'RTS')

        plt.title(self.modelName + ":" +"Histogram [dB]",fontsize=fontSize)
        plt.legend(fontsize=fontSize)
        plt.xlabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.ylabel('Percentage', fontsize=fontSize)
        plt.tick_params(labelsize=fontSize)
        plt.grid(True)
        plt.savefig(fileName)


    def NNPlot_epochs_KF_RTS(self, N_MiniBatchTrain_plt, BatchSize, MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
                      MSE_KNet_test_dB_avg, MSE_KNet_cv_dB_epoch, MSE_KNet_train_dB_epoch,
                      MSE_RTSNet_test_dB_avg, MSE_RTSNet_cv_dB_epoch, MSE_RTSNet_train_dB_epoch):
        N_Epochs_plt = np.floor(N_MiniBatchTrain_plt/BatchSize).astype(int) # number of epochs

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train KNet and RTSNet
        # y_plt1 = MSE_KNet_train_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        # plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])
        # y_plt2 = MSE_RTSNet_train_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        # plt.plot(x_plt, y_plt2, color=RTSColor[0],linestyle='-', marker='o', label=ERTSlegend[0])

        # CV KNet and RTSNet
        y_plt3 = MSE_KNet_cv_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt3, color=RTSColor[0],linestyle='-', marker='o', label=Klegend[1])
        y_plt4 = MSE_RTSNet_cv_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt4, color=RTSColor[1],linestyle='-', marker='o', label=ERTSlegend[1])

        # Test KNet and RTSNet
        y_plt5 = MSE_KNet_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, color=RTSColor[0],linestyle='--', label=Klegend[2])
        y_plt6 = MSE_RTSNet_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt6,color=RTSColor[1],linestyle='--', label=ERTSlegend[2])

        # RTS
        y_plt7 = MSE_ERTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt7, RTSColor[2], label=ERTSlegend[3])

        # EKF
        y_plt8 = MSE_EKF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt8, RTSColor[3], label=ERTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.grid(True)
        plt.savefig(fileName)

    def plotTrajectories(self,inputs, dim, titles, file_name):

        fig = plt.figure(figsize=(15,10))
        plt.Axes (fig, [0,0,1,1])
        # plt.subplots_adjust(wspace=-0.2, hspace=-0.2)
        print(len(inputs))
        matrix_size = int(np.ceil(np.sqrt(len(inputs))))
        #gs1 = gridspec.GridSpec(matrix_size,matrix_size)
        gs1 = gridspec.GridSpec(3,2)
        gs2 = gridspec.GridSpec(5,1)
        gs2.update(wspace=0, hspace=1)
        plt.rcParams["figure.frameon"] = False
        plt.rcParams["figure.constrained_layout.use"]= True
        i=0
        for title in titles:
            inputs_numpy = inputs[i].detach().numpy()
            gs1.update(wspace=0,hspace=0)
            if(dim==3):
                plt.rcParams["figure.frameon"] = False
                ax = fig.add_subplot(gs1[i],projection='3d')
                # if(i<3):
                #     ax = fig.add_subplot(gs1[i],projection='3d')
                # else:
                #     ax = fig.add_subplot(gs1[i:i+2],projection='3d')

                y_al = 0.8
                if(title == "True Trajectory"):
                    c = 'k'
                elif(title == "Observation"):
                    c = 'r'
                elif(title == "Extended RTS" or title =="Extended Kalman Filter"):
                    c = 'b'
                elif(title == "RTSNet" or title =="KalmanNet"):
                    c = 'g'
                else:
                    c = 'm'
                    # y_al = 0.68

                ax.set_axis_off()
                ax.set_title(title, y=y_al, fontdict={'fontsize': 15,'fontweight' : 20,'verticalalignment': 'baseline'})
                ax.plot(inputs_numpy[0,0,:], inputs_numpy[0,1,:], inputs_numpy[0,2,:], c, linewidth=0.5)

                ## Plot display
                #ax.set_yticklabels([])
                #ax.set_xticklabels([])
                #ax.set_zticklabels([])
                #ax.set_xlabel('x')
                #ax.set_ylabel('y')
                #ax.set_zlabel('z')

            if(dim==2):
                ax = fig.add_subplot(matrix_size, matrix_size,i+1)
                ax.plot(inputs_numpy[0,0,:],inputs_numpy[0,1,:], 'b', linewidth=0.75)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_title(title, pad=10, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})

            if(dim==4):
                if(title == "True Trajectory"):
                    target_theta_sample = inputs_numpy[0,0,:]

                # ax = fig.add_subplot(matrix_size, matrix_size,i+1)
                ax = fig.add_subplot(gs2[i,:])
                # print(inputs_numpy[0,0,:])
                ax.plot(np.arange(np.size(inputs_numpy[0,:],axis=1)), inputs_numpy[0,0,:], 'b', linewidth=0.75)
                if(title != "True Trajectory"):
                    diff = target_theta_sample - inputs_numpy[0,0,:]
                    peaks, _ = find_peaks(diff, prominence=0.31)
                    troughs, _ = find_peaks(-diff, prominence=0.31)
                    for peak, trough in zip(peaks, troughs):
                        plt.axvspan(peak, trough, color='red', alpha=.2)
                # zoomed in
                # ax.plot(np.arange(20), inputs_numpy[0,0,0:20], 'b', linewidth=0.75)inputs_numpy[0,0,:]
                ax.set_xlabel('time [s]')
                ax.set_ylabel('theta [rad]')
                ax.set_title(title, pad=10, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})

            i +=1
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=1000)

    def Partial_Plot_Lor(self, r, MSE_Partial_dB):
        fileName = self.folderName + 'Nonlinear_Lor_Partial_J=2'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12, label=r'EKF:  $\rm J_{mdl}=5$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12, label=r'EKF:  $\rm J_{mdl}=2$')
        main_partial.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12, label=r'RTS:  $\rm J_{mdl}=5$')
        main_partial.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12, label=r'RTS:  $ \rm J_{mdl}=2$')
        main_partial.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12, label=r'RTSNet: $ \rm J_{mdl}=2$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-60, 10))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)

        ax2 = plt.axes([.15, .15, .25, .25])
        x1, x2, y1, y2 =  19.5, 20.5, -35, -10
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12)
        ax2.grid(True)
        plt.savefig(fileName)

    def Partial_Plot_Pen(self, r, MSE_Partial_dB):
        fileName = self.folderName + 'Nonlinear_Pen_PartialF'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=4, markersize=12, label=r'EKF:  $\rm L=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=4, markersize=12, label=r'EKF:  $\rm L=1.1$')
        main_partial.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'RTS:  $\rm L=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12, label=r'RTS:  $ \rm L=1.1$')
        main_partial.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=24, label=r'RTSNet: $ \rm L=1.1$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-75, 5))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)

        ax2 = plt.axes([.15, .15, .25, .25])
        x1, x2, y1, y2 =  19.5, 20.5, -55, -15
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12)
        ax2.grid(True)
        plt.savefig(fileName)

    def Partial_Plot_H1(self, r, MSE_Partial_dB):
        fileName = self.folderName + 'Nonlinear_Lor_Partial_Hrot1'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12, label=r'EKF:  $\Delta{\theta}=0$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12, label=r'EKF:  $\Delta{\theta}=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12, label=r'RTS:  $\Delta{\theta}=0$')
        main_partial.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12, label=r'RTS:  $\Delta{\theta}=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12, label=r'RTSNet: $\Delta{\theta}=1$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-60, 10))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)

        ax2 = plt.axes([.15, .15, .25, .25])
        x1, x2, y1, y2 =  19.5, 20.5, -35, -10
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12)
        ax2.grid(True)
        plt.savefig(fileName)

    def Partial_Plot_KNetRTSNet_Compare(self, r, MSE_Partial_dB):
        fileName = self.folderName + 'Nonlinear_Lor_Partial_Hrot1_Compare'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '--bo', linewidth=3, markersize=12, label=r'KNet: $\Delta{\theta}=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--g^', linewidth=3, markersize=12, label=r'RTSNet: $\Delta{\theta}=1$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-60, 10))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        plt.savefig(fileName)


