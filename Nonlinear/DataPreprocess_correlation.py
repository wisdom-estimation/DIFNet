import torch
from Extended_sysmdl_multisensor import SystemModel
from Utils.Extended_data import DataGen
from Utils.Sysmdl_Parameter import Model_s4, Model_2DCT, Model_3DCT

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

model = Model_3DCT()
n_list = model.n_list

state_idx = [0, 2, 4]
velocity_idx = [1, 3, 5]

initial_state_all_training = torch.zeros([model.N_E, 6, 1])
initial_state_all_validation = torch.zeros([model.N_CV, 6, 1])
initial_state_all_test = torch.zeros([model.N_T, 6, 1])
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
    model.f,
    model.Q,
    model.h1,
    model.h_list,
    model.n_list,
    model.R_mul,
    model.Trans,
    model.T,
    model.T_test,
    model.sensor_num,
)
sys_model.InitSequence(model.x0, model.P0, initial_state_all)
dataFolderName = "../" + "Simulations/NonLinear_Data" + "/"
dataFileName = "CT_6x6_sen4_T50.pt"
dataFileName_mix = "CT_6x6_sen4_T50_mix.pt"
print("Start Data Gen")
model.T = 50
DataGen(sys_model, dataFolderName + dataFileName_mix, model.T, model.T)
