import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
# from Linear_sysmdl_multisensor import SystemModel
from var_sysmdl import SystemModel
from Utils.Extended_data import DataGen, DataLoader_GPU, DataGen_var
from Utils.Sysmdl_Parameter import Model_s4
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

########################################
########### Model Coupled Mass #########
model = Model_s4()
n_list = model.n_list

state_idx = [0, 2, 4]
velocity_idx = [1, 3, 5]

initial_state_all_training = torch.zeros([model.N_E, 6, 1]) * 100
initial_state_all_validation = torch.zeros([model.N_CV, 6, 1]) * 100
initial_state_all_test = torch.zeros([model.N_T, 6, 1]) * 100
initial_state_all = {
    "training": initial_state_all_training,
    "validation": initial_state_all_validation,
    "test": initial_state_all_test,
}
for index in velocity_idx:
    initial_state_all_training[:, index, :] = 100
    initial_state_all_validation[:, index, :] = 100
    initial_state_all_test[:, index, :] = 100

sys_model = SystemModel(
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
sys_model.InitSequence(model.x0, model.P0, initial_state_all)
dataFolderName = "../" + "Simulations/Linear_Data" + "/"
dataFileName = "var_0.1_CV_6x6_sen4_T100.pt"
dataFileName_mix = "var_0.1_CV_6x6_sen4_T100_mix.pt"
print("Start Data Gen")
model.T = 50
DataGen_var(sys_model, dataFolderName + dataFileName_mix, model.T, model.T, 0.7)
