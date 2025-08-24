import torch

from var_sysmdl import SystemModel
from Utils.Extended_data import DataLoader_GPU, My_Dataset, DataGen_var
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

model = Model_s4()

# sigma_list = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
sigma_list = [0.1, 0.3, 0.5, 0.7, 0.9]
sigma_length = len(sigma_list)
DIF_exact_RMSE_pos = [torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)]
DIF_exact_RMSE_vel = [torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)]
DIF_inexact_RMSE_pos = [
    torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)
]
DIF_inexact_RMSE_vel = [
    torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)
]
CKF_RMSE_pos = [torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)]
CKF_RMSE_vel = [torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)]
DIFNet_RMSE_pos = [torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)]
DIFNet_RMSE_vel = [torch.zeros(size=[model.sensor_num]) for i in range(sigma_length)]

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

state_idx = [0, 2, 4]
velocity_idx = [1, 3, 5]
initial_state_all_training = torch.zeros([model.N_E, 6, 1]) * 100
initial_state_all_validation = torch.zeros([model.N_CV, 6, 1]) * 100
initial_state_all_test = torch.ones([model.N_T, 6, 1]) * 100
initial_state_all = {
    "training": initial_state_all_training,
    "validation": initial_state_all_validation,
    "test": initial_state_all_test,
}
for index in velocity_idx:
    initial_state_all_training[:, index, :] = 100
    initial_state_all_validation[:, index, :] = 100
    initial_state_all_test[:, index, :] = 100

sys_model_DataGen = SystemModel(
    model.F_cv,
    model.Q_cv,
    model.H_mul,
    model.H_mul,
    model.H_list,
    model.R_mul,
    model.Trans,
    model.T,
    model.T_test,
    model.sensor_num,
)
sys_model_DataGen.InitSequence(model.x0, model.P0, initial_state_all)
dataFolderName = "../" + "Simulations/Linear_Data" + "/"

sys_model_exact = SystemModel(
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
sys_model_exact.InitSequence(model.filter_x0, model.filter_P0, initial_state_all)

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
        model.sensor_num,
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
        model.sensor_num,
    )
    for i in range(model.sensor_num)
]

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
    model.sensor_num,
)
sys_model_CKF.InitSequence(model.filter_x0, model.filter_P0, initial_state_all)


# for i in range(sigma_length):
#     dataFileName = f"var_{sigma_list[i]}_CV_sen4_T100.pt"
#     dataFileName_mix = f"var_{sigma_list[i]}_CV_sen4_T100_mix.pt"
#     print(f"Start Data Gen sigma={sigma_list[i]}")
#     model.T = 50
#     DataGen_var(
#         sys_model_DataGen,
#         dataFolderName + dataFileName_mix,
#         model.T,
#         model.T,
#         sigma_list[i],
#     )
#     print(f"Data Gen Done sigma={sigma_list[i]}")

#     print(f"Mix Data Load sigma={sigma_list[i]}")
#     [
#         train_input_mix,
#         train_target,
#         cv_input_mix,
#         cv_target,
#         test_input_mix,
#         test_target,
#     ] = DataLoader_GPU(dataFolderName + dataFileName_mix)

#     # 构建数据加载器
#     train_dataset = My_Dataset(
#         train_input_mix, train_target, initial_state_all["training"]
#     )
#     valid_dataset = My_Dataset(cv_input_mix, cv_target, initial_state_all["validation"])
#     test_dataset = My_Dataset(test_input_mix, test_target, initial_state_all["test"])

#     train_loader = DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
#     )
#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
#     )

#     position_idx = [[0, 2], [0, 2], [0, 2, 4], [0]]
#     velocity_idx = [[1, 3], [1, 3], [1, 3, 5], [1]]

#     ###################################
#     ### Evaluate Information Filter ###
#     ##################################
#     print(f"Evaluate Information Filter Exact sigma={sigma_list[i]}")
#     [
#         MSE_DIF_linear_arr,
#         MSE_DIF_linear_avg,
#         MSE_DIF_dB_avg,
#         DIF_out,
#         DIF_MSE_seq,
#         DIF_MSE_seq_pos,
#         DIF_MSE_seq_vel,
#     ] = DIFTest(
#         sys_model_exact,
#         sys_model_kf_exact,
#         test_input_mix,
#         test_target,
#         position_idx,
#         velocity_idx,
#         model.sensor_num,
#     )
#     print(f"Evaluate Information Filter Inexact sigma={sigma_list[i]}")
#     [
#         MSE_DIF_inexact_linear_arr,
#         MSE_DIF_inexact_linear_avg,
#         MSE_DIF_inexact_dB_avg,
#         DIF_inexact_out,
#         DIF_inexact_MSE_seq,
#         DIF_inexact_MSE_seq_pos,
#         DIF_inexact_MSE_seq_vel,
#     ] = DIFTest(
#         sys_model_inexact,
#         sys_model_kf_inexact,
#         test_input_mix,
#         test_target,
#         position_idx,
#         velocity_idx,
#         model.sensor_num,
#     )

#     ################################
#     ### Centralized Kalman Filter###
#     ################################
#     print(f"Evaluate Centralized Kalman Filter sigma={sigma_list[i]}")
#     [
#         MSE_KF_linear_arr,
#         MSE_KF_linear_avg,
#         MSE_KF_dB_avg,
#         KF_out,
#         KF_MSE_seq,
#         KF_MSE_seq_pos,
#         KF_MSE_seq_vel,
#     ] = KFTest(
#         sys_model_CKF,
#         test_input_mix,
#         test_target,
#         model.sensor_num,
#         model.Trans,
#         position_idx,
#         velocity_idx,
#     )

#     print(f"Start DIFNet pipeline sigma={sigma_list[i]}")
#     modelFolder = "../" + "Decentralized IFNet" + "/"
#     modelStateDictFolder = modelFolder + "model_state_dict/"
#     IFNet_Pipeline = Pipeline_IF(
#         strTime, modelFolder, f"var_{sigma_list[i]}_CV_DIFNet_sen4"
#     )
#     IFNet_Pipeline.setssModel(sys_model_inexact, sys_model_kf_inexact, args.batch_size)
#     IFNet_model = Decentralized_IFNetNN()
#     IFNet_model.Build(sys_model_inexact, sys_model_kf_inexact, args.batch_size)
#     IFNet_Pipeline.setModel(IFNet_model)
#     IFNet_Pipeline.setTrainingParams(n_Epochs=100, learningRate=1e-3, weightDecay=1e-4)

#     # IFNet_Pipeline.NNTrain(model.N_E, train_loader, valid_loader, model.N_CV)
#     [
#         IFNet_MSE_test_linear_arr,
#         IFNet_MSE_test_linear_avg,
#         IFNet_MSE_test_dB_avg,
#         IFNet_test,
#         IFNet_MSE_seq,
#         IFNet_MSE_seq_pos,
#         IFNet_MSE_seq_vel,
#     ] = IFNet_Pipeline.NNTest(model.N_T, test_loader, position_idx, velocity_idx)

#     for j in range(model.sensor_num):
#         DIF_exact_RMSE_pos[i][j] = torch.sqrt(torch.mean(DIF_MSE_seq_pos[j] ** 2))
#         DIF_exact_RMSE_vel[i][j] = torch.sqrt(torch.mean(DIF_MSE_seq_vel[j] ** 2))

#         DIF_inexact_RMSE_pos[i][j] = torch.sqrt(
#             torch.mean(DIF_inexact_MSE_seq_pos[j] ** 2)
#         )
#         DIF_inexact_RMSE_vel[i][j] = torch.sqrt(
#             torch.mean(DIF_inexact_MSE_seq_vel[j] ** 2)
#         )

#         CKF_RMSE_pos[i][j] = torch.sqrt(torch.mean(KF_MSE_seq_pos[j] ** 2))
#         CKF_RMSE_vel[i][j] = torch.sqrt(torch.mean(KF_MSE_seq_vel[j] ** 2))

#         DIFNet_RMSE_pos[i][j] = torch.sqrt(torch.mean(IFNet_MSE_seq_pos[j] ** 2))
#         DIFNet_RMSE_vel[i][j] = torch.sqrt(torch.mean(IFNet_MSE_seq_vel[j] ** 2))

# torch.save(
#     [
#         DIF_exact_RMSE_pos,
#         DIF_exact_RMSE_vel,
#         DIF_inexact_RMSE_pos,
#         DIF_inexact_RMSE_vel,
#         CKF_RMSE_pos,
#         CKF_RMSE_vel,
#         DIFNet_RMSE_pos,
#         DIFNet_RMSE_vel,
#     ],
#     "./time_var_sen4_RMSE.pt",
# )


[
    DIF_exact_RMSE_pos,
    DIF_exact_RMSE_vel,
    DIF_inexact_RMSE_pos,
    DIF_inexact_RMSE_vel,
    CKF_RMSE_pos,
    CKF_RMSE_vel,
    DIFNet_RMSE_pos,
    DIFNet_RMSE_vel,
] = torch.load("./time_var_sen4_RMSE.pt", weights_only=True)

plt.figure(figsize=[12, 8])
for i in range(model.sensor_num):
    ax = plt.subplot(2, 2, i + 1)
    ax.plot(
        sigma_list,
        [k[i].cpu().numpy() for k in DIF_exact_RMSE_pos],
        label=f"sensor {i+1} DIF_exact",
        linewidth=0.4,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax.plot(
        sigma_list,
        [k[i].detach().cpu().numpy() for k in DIF_inexact_RMSE_pos],
        label=f"sensor {i+1} DIF_inexact",
        linewidth=0.4,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax.plot(
        sigma_list,
        [k[i].detach().cpu().numpy() for k in CKF_RMSE_pos],
        label="Centralized KF",
        linewidth=0.4,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax.plot(
        sigma_list,
        [k[i].detach().cpu().numpy() for k in DIFNet_RMSE_pos],
        label=f"sensor {i+1} Decentralized IFNet",
        linewidth=0.4,
        color="red",
        marker="p",
        markersize=5,
    )
    ax.set_xlabel(r"$\sigma$", fontsize=13)
    ax.set_ylabel("RMSE of position (m)")
    ax.legend()
plt.savefig("var_linear_s4_pos_rmse.pdf", bbox_inches="tight", dpi=300, format="pdf")

plt.figure(figsize=[12, 8])
for i in range(model.sensor_num):
    ax = plt.subplot(2, 2, i + 1)
    ax.plot(
        sigma_list,
        [k[i].detach().cpu().numpy() for k in DIF_exact_RMSE_vel],
        label=f"sensor {i+1} DIF_exact",
        linewidth=0.4,
        color="pink",
        marker="o",
        markersize=7,
    )
    ax.plot(
        sigma_list,
        [k[i].detach().cpu().numpy() for k in DIF_inexact_RMSE_vel],
        label=f"sensor {i+1} DIF_inexact",
        linewidth=0.4,
        color="orange",
        marker="v",
        markersize=5,
    )
    ax.plot(
        sigma_list,
        [k[i].detach().cpu().numpy() for k in CKF_RMSE_vel],
        label="Centralized KF",
        linewidth=0.4,
        color="blue",
        marker="*",
        markersize=5,
    )
    ax.plot(
        sigma_list,
        [k[i].detach().cpu().numpy() for k in DIFNet_RMSE_vel],
        label=f"sensor {i+1} Decentralized IFNet",
        linewidth=0.4,
        color="red",
        marker="p",
        markersize=5,
    )
    ax.set_xlabel(r"$\sigma$", fontsize=13)
    ax.set_ylabel("RMSE of velocity (m/s)")
    ax.legend()
plt.savefig("var_linear_s4_vel_rmse.pdf", bbox_inches="tight", dpi=300, format="pdf")

plt.show()
