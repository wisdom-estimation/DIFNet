import torch

from Linear_sysmdl_multisensor import SystemModel
from Utils.Extended_data import DataLoader_GPU, log_record, My_Dataset, DataLoader_CPU
from Utils.Sysmdl_Parameter import Model_s4
from Pipeline_IF import Pipeline_IF
from IFNet_nn import Decentralized_IFNetNN
from KalmanFilter_test import KFTest
from DIF_test import DIFTest
from datetime import datetime
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

# 获取外部配置参数
parser = argparse.ArgumentParser(description="A simple command-line parameter example")
# 数据配置
parser.add_argument(
    "--Data_folder", type=str, default="../Simulations/Linear_Data/", required=False
)
parser.add_argument(
    "--dataFileName", type=str, default="CV_4x2_r100r200q1_sen2_T100.pt", required=False
)
parser.add_argument(
    "--dataFileName_mix",
    type=str,
    default="CV_4x2_r100r200q1_sen2_T100_mix.pt",
    required=False,
)

# 训练参数设置W
parser.add_argument("--batch_size", type=int, default=20, required=False)
parser.add_argument("--train_epoch", type=int, default=500, required=False)
parser.add_argument("--learning_rate", type=float, default=0.001, required=False)
parser.add_argument("--weight_decay", type=float, default=0.0001, required=False)
# 网络消融参数
parser.add_argument("--Net_layers", type=int, default=3, required=False)  # 没有接入网络
args = parser.parse_args()


def main(args):
    sensor_num = 4
    model = Model_s4()
    model.sensor_num = sensor_num

    golbal_position_idx = [0, 2, 4]
    golbal_velocity_idx = [1, 3, 5]

    # dis = MultivariateNormal(loc=model.x0.squeeze(), covariance_matrix=model.P0)
    # initial_state = dis.rsample().unsqueeze(-1)

    # initial_state_all_training = initial_state.repeat(model.N_E, 1, 1)
    # initial_state_all_validation = initial_state.repeat(model.N_CV, 1, 1)
    # initial_state_all_test = initial_state.repeat(model.N_T, 1, 1)

    initial_state_all_training = torch.ones([model.N_E, 6, 1]) * 100
    initial_state_all_validation = torch.ones([model.N_CV, 6, 1]) * 100
    initial_state_all_test = torch.ones([model.N_T, 6, 1]) * 100
    initial_state_all = {
        "training": initial_state_all_training,
        "validation": initial_state_all_validation,
        "test": initial_state_all_test,
    }
    # for index in golbal_velocity_idx:
    #     initial_state_all_training[:, index, :] = 100
    #     initial_state_all_validation[:, index, :] = 100
    #     initial_state_all_test[:, index, :] = 100

    ## exact IF
    sys_model = SystemModel(
        model.F_cv,
        model.Q_cv,
        model.H_cv1,
        model.H_mul,
        model.H_list,
        model.R_mul,
        model.Trans,
        model.T,
        model.T_test,
        model.sensor_num,
    )
    sys_model.InitSequence(model.filter_x0, model.filter_P0, initial_state_all)

    ## inexact IF
    sys_model_inexact = SystemModel(
        model.filter_F_cv,
        model.filter_inexactIF_Q_cv,
        model.H_cv1,
        model.H_mul,
        model.H_list,
        model.filter_inexact_R_cv,
        model.Trans,
        model.T,
        model.T_test,
        model.sensor_num,
    )
    sys_model_inexact.InitSequence(model.filter_x0, model.filter_P0, initial_state_all)

    # Distributed sensor KF
    sys_model_kf_exact = [
        SystemModel(
            model.F_cv,
            model.Q_cv,
            model.H_cv1,
            model.H_mul,
            model.H_list,
            model.R_list[i],
            model.Trans,
            model.T,
            model.T_test,
            sensor_num,
        )
        for i in range(model.sensor_num)
    ]

    sys_model_kf_inexact = [
        SystemModel(
            model.filter_F_cv,
            model.filter_Q_cv,
            model.H_cv1,
            model.H_mul,
            model.H_list,
            model.R_list[i],
            model.Trans,
            model.T,
            model.T_test,
            sensor_num,
        )
        for i in range(sensor_num)
    ]

    # Centralized KF
    sys_model_CKF = SystemModel(
        model.F_cv,
        model.Q_cv,
        model.H_mul,
        model.H_mul,
        model.H_list,
        model.R_mul,
        model.Trans,
        model.T,
        model.T_test,
        sensor_num,
    )
    sys_model_CKF.InitSequence(model.filter_x0, model.filter_P0, initial_state_all)

    dataFolderName = "../" + "Simulations/Linear_Data" + "/"
    dataFileName_mix = "CV_6x6_sen4_T100_mix.pt"
    print("Mix data Load")
    if torch.cuda.is_available():
        [
            train_input_mix,
            train_target,
            cv_input_mix,
            cv_target,
            test_input_mix,
            test_target,
        ] = DataLoader_GPU(dataFolderName + dataFileName_mix)

    else:
        [
            train_input_mix,
            train_target,
            cv_input_mix,
            cv_target,
            test_input_mix,
            test_target,
        ] = DataLoader_CPU(dataFolderName + dataFileName_mix)

    # 构建数据加载器
    train_dataset = My_Dataset(
        train_input_mix, train_target, initial_state_all["training"]
    )
    valid_dataset = My_Dataset(cv_input_mix, cv_target, initial_state_all["validation"])
    test_dataset = My_Dataset(test_input_mix, test_target, initial_state_all["test"])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    # each sensor's local state space index
    position_idx = [[0, 2], [0, 2], [0, 2, 4], [0]]
    velocity_idx = [[1, 3], [1, 3], [1, 3, 5], [1]]

    ##################################
    ## Evaluate Information Filter ###
    #################################
    print("Evaluate Information Filter Exact")
    [
        MSE_DIF_linear_arr,
        MSE_DIF_linear_avg,
        MSE_DIF_dB_avg,
        DIF_out,
        DIF_MSE_seq,
        DIF_MSE_seq_pos,
        DIF_MSE_seq_vel,
        DIF_exact_time,
    ] = DIFTest(
        sys_model,
        sys_model_kf_exact,
        test_input_mix,
        test_target,
        position_idx,
        velocity_idx,
        sensor_num,
    )

    print("Evaluate Information Filter Inexact")
    [
        MSE_DIF_inexact_linear_arr,
        MSE_DIF_inexact_linear_avg,
        MSE_DIF_inexact_dB_avg,
        DIF_inexact_out,
        DIF_inexact_MSE_seq,
        DIF_inexact_MSE_seq_pos,
        DIF_inexact_MSE_seq_vel,
        DIF_inexact_time,
    ] = DIFTest(
        sys_model_inexact,
        sys_model_kf_inexact,
        test_input_mix,
        test_target,
        position_idx,
        velocity_idx,
        sensor_num,
    )

    ################################
    ### Centralized Kalman Filter###
    ################################
    print("Evaluate Centralized Kalman Filter ")
    [
        MSE_KF_linear_arr,
        MSE_KF_linear_avg,
        MSE_KF_dB_avg,
        KF_out,
        KF_MSE_seq,
        KF_MSE_seq_pos,
        KF_MSE_seq_vel,
        CKF_time,
    ] = KFTest(
        sys_model_CKF,
        test_input_mix,
        test_target,
        sensor_num,
        model.Trans,
        position_idx,
        velocity_idx,
    )

    ###############
    ###  IFNet ####
    ###############
    print("Start IFNet pipeline")
    modelFolder = "../" + "Decentralized IFNet" + "/"
    modelStateDictFolder = modelFolder + "model_state_dict/"
    IFNet_Pipeline = Pipeline_IF(strTime, modelFolder, "CV_DIFNet_sen4_v2")
    IFNet_Pipeline.setssModel(sys_model_inexact, sys_model_kf_inexact, args.batch_size)
    IFNet_model = Decentralized_IFNetNN()
    IFNet_model.Build(sys_model_inexact, sys_model_kf_inexact, args.batch_size)
    IFNet_Pipeline.setModel(IFNet_model)
    IFNet_Pipeline.setTrainingParams(n_Epochs=100, learningRate=1e-3, weightDecay=1e-4)

    # IFNet_Pipeline.NNTrain(model.N_E, train_loader, valid_loader, model.N_CV)
    # IFNet_Pipeline.save()
    [
        IFNet_MSE_test_linear_arr,
        IFNet_MSE_test_linear_avg,
        IFNet_MSE_test_dB_avg,
        IFNet_test,
        IFNet_MSE_seq,
        IFNet_MSE_seq_pos,
        IFNet_MSE_seq_vel,
        DIFNet_time,
    ] = IFNet_Pipeline.NNTest(model.N_T, test_loader, position_idx, velocity_idx)
    DIFNet_time *= args.batch_size

    ##Plot##
    T = 50
    plt.figure(figsize=[12, 8])
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        range(T),
        DIF_MSE_seq_pos[0].detach().cpu().numpy(),
        label="sensor 1 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax1.plot(
        range(T),
        DIF_inexact_MSE_seq_pos[0].detach().cpu().numpy(),
        label="sensor 1 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax1.plot(
        range(T),
        KF_MSE_seq_pos[0].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax1.plot(
        range(T),
        IFNet_MSE_seq_pos[0].detach().cpu().numpy(),
        label="sensor 1 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("RMSE of position (m)")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(
        range(T),
        DIF_MSE_seq_pos[1].detach().cpu().numpy(),
        label="sensor 2 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax2.plot(
        range(T),
        DIF_inexact_MSE_seq_pos[1].detach().cpu().numpy(),
        label="sensor 2 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax2.plot(
        range(T),
        KF_MSE_seq_pos[1].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax2.plot(
        range(T),
        IFNet_MSE_seq_pos[1].detach().cpu().numpy(),
        label="sensor 2 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("RMSE of position (m)")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(
        range(T),
        DIF_MSE_seq_pos[2].detach().cpu().numpy(),
        label="sensor 3 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax3.plot(
        range(T),
        DIF_inexact_MSE_seq_pos[2].detach().cpu().numpy(),
        label="sensor 3 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax3.plot(
        range(T),
        KF_MSE_seq_pos[2].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax3.plot(
        range(T),
        IFNet_MSE_seq_pos[2].detach().cpu().numpy(),
        label="sensor 3 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("RMSE of position (m)")
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(
        range(T),
        DIF_MSE_seq_pos[3].detach().cpu().numpy(),
        label="sensor 4 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax4.plot(
        range(T),
        DIF_inexact_MSE_seq_pos[3].detach().cpu().numpy(),
        label="sensor 4 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax4.plot(
        range(T),
        KF_MSE_seq_pos[3].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax4.plot(
        range(T),
        IFNet_MSE_seq_pos[3].detach().cpu().numpy(),
        label="sensor 4 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("RMSE of position (m)")
    ax4.legend()
    plt.savefig("linear_s4_pos_rmse.pdf", bbox_inches="tight", dpi=300, format="pdf")

    plt.figure(figsize=[12, 8])
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        range(T),
        DIF_MSE_seq_vel[0].detach().cpu().numpy(),
        label="sensor 1 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax1.plot(
        range(T),
        DIF_inexact_MSE_seq_vel[0].detach().cpu().numpy(),
        label="sensor 1 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax1.plot(
        range(T),
        KF_MSE_seq_vel[0].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax1.plot(
        range(T),
        IFNet_MSE_seq_vel[0].detach().cpu().numpy(),
        label="sensor 1 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("RMSE of velocity (m/s)")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(
        range(T),
        DIF_MSE_seq_vel[1].detach().cpu().numpy(),
        label="sensor 2 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax2.plot(
        range(T),
        DIF_inexact_MSE_seq_vel[1].detach().cpu().numpy(),
        label="sensor 2 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax2.plot(
        range(T),
        KF_MSE_seq_vel[1].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax2.plot(
        range(T),
        IFNet_MSE_seq_vel[1].detach().cpu().numpy(),
        label="sensor 2 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("RMSE of velocity (m/s)")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(
        range(T),
        DIF_MSE_seq_vel[2].detach().cpu().numpy(),
        label="sensor 3 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax3.plot(
        range(T),
        DIF_inexact_MSE_seq_vel[2].detach().cpu().numpy(),
        label="sensor 3 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax3.plot(
        range(T),
        KF_MSE_seq_vel[2].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax3.plot(
        range(T),
        IFNet_MSE_seq_vel[2].detach().cpu().numpy(),
        label="sensor 3 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("RMSE of velocity (m/s)")
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(
        range(T),
        DIF_MSE_seq_vel[3].detach().cpu().numpy(),
        label="sensor 4 DIF_exact",
        linewidth=0.5,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax4.plot(
        range(T),
        DIF_inexact_MSE_seq_vel[3].detach().cpu().numpy(),
        label="sensor 4 DIF_inexact",
        linewidth=0.5,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax4.plot(
        range(T),
        KF_MSE_seq_vel[3].detach().cpu().numpy(),
        label="Centralized KF",
        linewidth=0.5,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax4.plot(
        range(T),
        IFNet_MSE_seq_vel[3].detach().cpu().numpy(),
        label="sensor 4 Decentralized IFNet",
        linewidth=0.5,
        color="red",
        marker="p",
        markersize=5,
    )
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("RMSE of velocity (m/s)")
    ax4.legend()

    plt.savefig("linear_s4_vel_rmse.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.show()

    print(
        torch.tensor([DIF_exact_time, DIF_inexact_time, CKF_time, DIFNet_time])
        / DIF_exact_time
    )


if __name__ == "__main__":
    main(args)
