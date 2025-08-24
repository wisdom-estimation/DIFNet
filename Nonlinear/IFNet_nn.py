import torch
import torch.nn as nn
from DEKF import DEKF
import torch.nn.init as init
from torch import autograd
import time


seed = 2003
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class IFNetNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def getJacobian(self, y, g):
        Jac = autograd.functional.jacobian(g, y)
        Jac = Jac.view(-1, self.ssmodel.m)
        return Jac

    def Build(self, ssModel, sensor_index, batch_size):
        self.ssmodel = ssModel
        self.sensor_num = ssModel.sensor_num
        self.sensor_index = sensor_index
        self.batch_size = batch_size

        self.InitSystemDynamics(ssModel)

        # Number of neurons in the 2nd net hidden layer
        H1_InfoNet = (ssModel.m**2 + ssModel.m**2) * 1 * self.sensor_num
        # Number of neurons in the 3nd net hidden layer
        H2_InfoNet = (ssModel.m**2) * 5

        self.InitInfoNet(H1_InfoNet, H2_InfoNet)

        self.Initial_Weights()

    def InitInfoNet(self, H1, H2):
        # Input Dimensions

        D_in = (
            self.m * self.sensor_num + self.m * self.m * self.sensor_num
        )  #  yi_k|k - yi_k|k-1, Pi_k|k - Pi_k|k-1,k

        # Output Dimensions

        D_out = self.m * self.m * self.sensor_num  # parameter

        ###################
        ### Input Layer ###
        ###################

        # Linear Layer
        self.Info_l1_in = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.Info_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m) * 10
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        # self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(
            self.device, non_blocking=True
        )

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Hidden Layer ###
        ####################
        self.Info_l2_in = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.Info_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.Info_l3_in = torch.nn.Linear(H2, D_out, bias=True)

    def Initial_Weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
                # init.xavier_normal_(m.weight)
                # init.constant_(m.bias, 0)

        for name, param in self.rnn_GRU.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.xavier_uniform_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def InitSystemDynamics(self, ssModel):
        self.f = ssModel.f
        self.m = ssModel.m

        self.h_list = ssModel.h_list
        self.n_list = ssModel.n_list

        self.Q = ssModel.Q

        self.Trans = [
            ssModel.Trans[i]
            .repeat(self.batch_size, 1, 1)
            .to(self.device, non_blocking=True)
            for i in range(self.sensor_num)
        ]
        self.Trans_Q = []
        for i in range(self.sensor_num):
            self.Trans_Q.append(self.Trans[i] @ self.Q @ self.Trans[i].transpose(1, 2))

    def InitSequence(self, M1_0, M2_0, initial_state):
        self.m1x_prior = M1_0.to(self.device, non_blocking=True)
        self.m2x_prior = M2_0.to(self.device, non_blocking=True)

        self.m1x_posterior = self.Trans[self.sensor_index] @ initial_state
        self.m2x_posterior = (
            self.Trans[self.sensor_index]
            @ M2_0
            @ self.Trans[self.sensor_index].transpose(1, 2)
        )

    def Predict(self):
        self.m1x_prior = torch.empty(
            size=[self.batch_size, self.ssmodel.Trans[self.sensor_index].size()[0], 1]
        )
        for i in range(self.batch_size):
            self.m1x_prior[i, :, :] = self.ssmodel.Trans[self.sensor_index] @ self.f(
                torch.pinverse(self.ssmodel.Trans[self.sensor_index])
                @ self.m1x_posterior[i, :, :]
            )
        self.F = torch.empty(
            size=[
                self.batch_size,
                self.Trans[self.sensor_index].size()[1],
                self.Trans[self.sensor_index].size()[1],
            ]
        )
        for i in range(self.batch_size):
            self.F[i, :, :] = (
                self.ssmodel.Trans[self.sensor_index]
                @ self.getJacobian(
                    torch.pinverse(self.ssmodel.Trans[self.sensor_index])
                    @ self.m1x_posterior[i, :, :],
                    self.f,
                )
                @ torch.pinverse(self.ssmodel.Trans[self.sensor_index])
            )
        self.m2x_prior = (
            self.F @ self.m2x_posterior @ self.F.transpose(1, 2)
            + self.Trans_Q[self.sensor_index]
        )

        self.Info = torch.inverse(self.m2x_prior)
        self.info = self.Info @ self.m1x_prior

    def Estimation(self, info_in):
        ###################
        ### Input Layer ###
        ###################
        L1_out = self.Info_l1_in(info_in)
        La1_out = self.Info_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            non_blocking=True,
        )
        GRU_in[0, :, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.batch_size, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.Info_l2_in(GRU_out_reshape)
        La2_out = self.Info_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.Info_l3_in(La2_out)
        L3_out_reshape = torch.reshape(
            L3_out, (self.batch_size, self.sensor_num, self.m, self.m)
        )

        return L3_out_reshape

    def Update(
        self, x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m
    ):
        self.time_cost = time.perf_counter()
        info_combination = torch.zeros(self.batch_size, self.sensor_num, self.m, 1)
        Info_combination = torch.zeros(self.batch_size, self.sensor_num, self.m, self.m)
        for i in range(self.sensor_num):
            if communication_m[self.sensor_index, i]:
                info_combination[:, i, 0 : self.Trans[i].shape[1], :] = (
                    torch.inverse(sigma_posterior[i][:, :, :]) @ x_posterior[i][:, :, :]
                    - torch.inverse(sigma_prior[i][:, :, :]) @ x_prior[i][:, :, :]
                )
                Info_combination[
                    :, i, 0 : self.Trans[i].shape[1], 0 : self.Trans[i].shape[1]
                ] = torch.inverse(sigma_posterior[i][:, :, :]) - torch.inverse(
                    sigma_prior[i][:, :, :]
                )
        info_combination_reshape = torch.reshape(
            info_combination, (self.batch_size, -1)
        )
        Info_combination_reshaoe = torch.reshape(
            Info_combination, (self.batch_size, -1)
        )

        info_in = torch.cat([info_combination_reshape, Info_combination_reshaoe], dim=1)
        Parameter = self.Estimation(info_in)
        self.Parameter = Parameter

        Info_est = self.Info
        info_est = self.info

        for i in range(self.sensor_num):
            if communication_m[self.sensor_index, i]:
                fusion_weight = (
                    torch.pinverse(self.Trans[self.sensor_index]).transpose(1, 2)
                    @ Parameter[:, i, :, :]
                    @ self.Trans[self.sensor_index].transpose(1, 2)
                )
                Info_est = Info_est + fusion_weight @ (
                    self.Trans[i] @ torch.pinverse(self.Trans[self.sensor_index])
                ).transpose(1, 2) @ (
                    torch.inverse(sigma_posterior[i][:, :, :])
                    - torch.inverse(sigma_prior[i][:, :, :])
                ) @ self.Trans[
                    i
                ] @ torch.pinverse(
                    self.Trans[self.sensor_index]
                )
                info_est = info_est + fusion_weight @ (
                    self.Trans[i] @ torch.pinverse(self.Trans[self.sensor_index])
                ).transpose(1, 2) @ (
                    torch.inverse(sigma_posterior[i][:, :, :]) @ x_posterior[i][:, :, :]
                    - torch.inverse(sigma_prior[i][:, :, :]) @ x_prior[i][:, :, :]
                )
        self.m2x_posterior = torch.inverse(Info_est)
        self.m1x_posterior = self.m2x_posterior @ info_est
        self.time_cost = time.perf_counter() - self.time_cost

    def Fusion(
        self, x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m
    ):
        # Predict y_k|k-1, Y_k|k-1
        self.Predict()
        # Update x_k|k
        self.Update(x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m)

    def forward(
        self, x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m
    ):
        self.Fusion(x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m)
        return (
            torch.squeeze(self.m1x_posterior),
            self.m2x_posterior,
            self.Parameter,
            self.time_cost,
        )

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data


class Decentralized_IFNetNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def Build(self, ssModel, ssModel_KF, batch_size):
        self.sensor_num = ssModel.sensor_num
        self.batch_size = batch_size
        self.NN_list = []
        self.ssModel = ssModel
        self.Trans = [
            ssModel.Trans[i]
            .repeat(self.batch_size, 1, 1)
            .to(self.device, non_blocking=True)
            for i in range(self.sensor_num)
        ]
        # 计算通信矩阵，判断传感器之间是否存在通信
        self.Communication_m = torch.eye(self.sensor_num)
        for i in range(self.sensor_num):
            for j in range(i + 1, self.sensor_num):
                if torch.any(
                    self.ssModel.Trans[j] @ torch.pinverse(self.ssModel.Trans[i])
                ):
                    self.Communication_m[i, j] = 1
                    self.Communication_m[j, i] = 1

        for i in range(self.sensor_num):
            model = IFNetNN()
            model.Build(ssModel, i, batch_size)
            self.NN_list.append(model)

        model_state_dict = self.NN_list[0].state_dict()
        for i in range(self.sensor_num):
            self.NN_list[i].load_state_dict(model_state_dict, strict=True)

    def InitSequence(self, m1x_0, m2x_0, initial_state):
        for i in range(self.sensor_num):
            self.NN_list[i].m1x_posterior = self.Trans[i] @ initial_state
            self.NN_list[i].m2x_posterior = (
                self.Trans[i] @ m2x_0 @ self.Trans[i].transpose(1, 2)
            )

    def Fusion(self, x_prior, x_posterior, sigma_prior, sigma_posterior):
        self.m1x_posterior = []
        self.m2x_posterior = []
        self.Parameter = []
        self.time_cost = []
        for i in range(self.sensor_num):
            # Local IFNet process
            x, sigma, Parameter, time_cost = self.NN_list[i](
                x_prior, x_posterior, sigma_prior, sigma_posterior, self.Communication_m
            )
            self.m1x_posterior.append(torch.pinverse(self.Trans[i]) @ x.unsqueeze(-1))
            self.m2x_posterior.append(
                torch.pinverse(self.Trans[i])
                @ sigma
                @ torch.pinverse(self.Trans[i]).transpose(1, 2)
            )
            self.Parameter.append(Parameter)
            self.time_cost.append(time_cost)

    def forward(self, x_prior, x_posterior, sigma_prior, sigma_posterior):
        self.Fusion(x_prior, x_posterior, sigma_prior, sigma_posterior)
        return self.m1x_posterior, self.m2x_posterior, self.Parameter, self.time_cost
