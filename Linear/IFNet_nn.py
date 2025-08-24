"""# **Class: IFNet**"""

import torch
import torch.nn as nn
import time
from KF import KalmanFilter
import torch.nn.init as init
from torch.distributions.multivariate_normal import MultivariateNormal


seed = 2023
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子


class IFNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #############
    ### Build ###
    #############
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

    ######################################
    ### Initialize Information Network ###
    ######################################
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

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, ssModel):
        # Set State Evolution Matrix
        self.F = ssModel.F.repeat(self.batch_size, 1, 1).to(
            self.device, non_blocking=True
        )
        self.F_T = torch.transpose(self.F, 1, 2)
        self.m = self.F.size()[1]
        self.Q = ssModel.Q

        # Set Observation Matrix
        self.H = ssModel.H.repeat(self.batch_size, 1, 1).to(
            self.device, non_blocking=True
        )
        self.H_T = torch.transpose(self.H, 1, 2)
        self.n = self.H.size()[1]
        self.R = ssModel.R

        # Set transformation information
        self.Trans = [
            ssModel.Trans[i]
            .repeat(self.batch_size, 1, 1)
            .to(self.device, non_blocking=True)
            for i in range(self.sensor_num)
        ]
        self.Trans_F = []
        self.Trans_Q = []
        for i in range(self.sensor_num):
            self.Trans_F.append(self.Trans[i] @ self.F @ torch.pinverse(self.Trans[i]))
            self.Trans_Q.append(self.Trans[i] @ self.Q @ self.Trans[i].transpose(1, 2))

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, M2_0, initial_state):

        self.m1x_prior = M1_0.to(self.device, non_blocking=True)
        self.m2x_prior = M2_0.to(self.device, non_blocking=True)

        self.m1x_posterior = self.Trans[self.sensor_index] @ initial_state
        self.m2x_posterior = (
            self.Trans[self.sensor_index]
            @ M2_0
            @ self.Trans[self.sensor_index].transpose(1, 2)
        )

    ####################
    ### Predict Step ###
    ####################
    def Predict(self, x_prior, sigma_prior):
        with torch.no_grad():
            # # Predict the 1-st moment of x
            # self.m1x_prior = self.Trans_F[self.sensor_index] @ self.m1x_posterior
            # # Predict the 2-nd moment of x
            # self.m2x_prior = torch.matmul(self.Trans_F[self.sensor_index], self.m2x_posterior)
            # self.m2x_prior = torch.matmul(self.m2x_prior, self.Trans_F[self.sensor_index].t()) + self.Trans[self.sensor_index] @ self.Q @ self.Trans[self.sensor_index].t()

            # self.m1x_prior = x_prior[self.sensor_index]
            # self.m2x_prior = sigma_prior[self.sensor_index]
            self.m1x_prior = self.Trans_F[self.sensor_index] @ self.m1x_posterior
            self.m2x_prior = (
                self.Trans_F[self.sensor_index]
                @ self.m2x_posterior
                @ self.Trans_F[self.sensor_index].transpose(1, 2)
                + self.Trans_Q[self.sensor_index]
            )

            # Transform the 2-nd moment of x into information matrix
            self.Info = torch.inverse(self.m2x_prior)

            # Transform the 1-st moment of x into information vector
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

    ###################
    ### Update Step ###
    ###################
    def Update(
        self, x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m
    ):
        self.time_cost = time.perf_counter()
        #######################
        ## Estimate Parameter##
        #######################
        info_combination = torch.zeros(self.batch_size, self.sensor_num, self.m, 1)
        Info_combination = torch.zeros(self.batch_size, self.sensor_num, self.m, self.m)
        for i in range(self.sensor_num):
            # 判断sensor_index传感器与i传感器之间是否有通信
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
            # 不需要补0，因为初始化即为0张量
            # else:
            #     info_combination[:, i, 0 : self.Trans[i].shape[1], :] = (
            #         torch.zeros_like(
            #             info_combination[:, i, 0 : self.Trans[i].shape[1], :]
            #         )
            #     )
            #     Info_combination[
            #         :, i, 0 : self.Trans[i].shape[1], 0 : self.Trans[i].shape[1]
            #     ] = torch.zeors_like(
            #         Info_combination[
            #             :, i, 0 : self.Trans[i].shape[1], 0 : self.Trans[i].shape[1]
            #         ]
            #     )

        info_combination_reshape = torch.reshape(
            info_combination, (self.batch_size, -1)
        )
        Info_combination_reshape = torch.reshape(
            Info_combination, (self.batch_size, -1)
        )

        info_in = torch.cat([info_combination_reshape, Info_combination_reshape], dim=1)
        Parameter = self.Estimation(info_in)
        self.Parameter = Parameter
        ##############################
        ## Update 1st and 2nd Moment##
        ##############################
        Info_est = self.Info
        info_est = self.info

        # self.H_mul = self.ssmodel.H_mul
        # self.R = self.ssmodel.R
        # self.R_Inverse = torch.inverse(self.R)
        # self.Filters_R = (100**2) * torch.eye(6)
        # self.Filters_H = torch.eye(6)

        for i in range(self.sensor_num):
            if communication_m[self.sensor_index, i]:
                # Parameter[i,:,:] = self.H_mul.t() @ self.R_Inverse[:,self.n*i:self.n*(i+1)] @ self.Filters_R @ torch.pinverse(self.Filters_H)
                # Parameter[i, :, :] = torch.diag_embed(torch.diag(Parameter[i, :, :]))
                fusion_weight = (
                    torch.pinverse(self.Trans[self.sensor_index]).transpose(1, 2)
                    @ Parameter[:, i, :, :]
                    @ self.Trans[self.sensor_index].transpose(1, 2)
                    # @ (
                    #     self.Trans[i] @ torch.pinverse(self.Trans[self.sensor_index])
                    # ).transpose(1, 2)
                )
                # fusion_weight = torch.pinverse(self.Trans[self.sensor_index]).T  @ Parameter[i,:,:] @ Parameter[i,:,:].T @\
                #                 self.Trans[self.sensor_index].T @ (self.Trans[i] @ torch.pinverse(self.Trans[self.sensor_index])).T
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

    ###################
    ### Fusion Step ###
    ###################
    def Fusion(
        self, x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m
    ):
        # Predict y_k|k-1, Y_k|k-1
        self.Predict(x_prior, sigma_prior)
        # Update x_k|k
        self.Update(x_prior, x_posterior, sigma_prior, sigma_posterior, communication_m)

    ###############
    ### Forward ###
    ###############
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

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data


class Decentralized_IFNetNN(torch.nn.Module):
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #############
    ### Build ###
    #############
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
            # self.NN_list[i].InitSystemDynamics(ssModel)

        ###统一初始化系数####
        model_state_dict = self.NN_list[0].state_dict()
        for i in range(self.sensor_num):
            self.NN_list[i].load_state_dict(model_state_dict, strict=True)

    def InitSequence(self, m1x_0, m2x_0, initial_state):
        for i in range(self.sensor_num):
            # self.NN_list[i].m1x_prior = self.Trans[i] @ initial_state
            self.NN_list[i].m1x_posterior = self.Trans[i] @ initial_state
            # self.NN_list[i].m2x_prior = (
            #     self.Trans[i] @ m2x_0 @ self.Trans[i].transpose(1, 2)
            # )
            self.NN_list[i].m2x_posterior = (
                self.Trans[i] @ m2x_0 @ self.Trans[i].transpose(1, 2)
            )

    ###################
    ### Fusion Step ###
    ###################
    def Fusion(self, x_prior, x_posterior, sigma_prior, sigma_posterior):
        self.m1x_posterior = []
        self.m2x_posterior = []
        self.Parameter = []
        self.time_cost = 0
        for i in range(self.sensor_num):
            # Local IFNet process
            x, sigma, Parameter, time_cost = self.NN_list[i](
                x_prior, x_posterior, sigma_prior, sigma_posterior, self.Communication_m
            )
            self.m1x_posterior.append(torch.pinverse(self.Trans[i]) @ x.unsqueeze(-1))
            # self.m2x_posterior.append(
            #     torch.pinverse(self.Trans[i])
            #     @ sigma
            #     @ torch.pinverse(self.Trans[i]).transpose(1, 2)
            # )
            self.m2x_posterior.append(sigma)
            self.Parameter.append(Parameter)
            self.time_cost += time_cost

    ###############
    ### Forward ###
    ###############
    def forward(self, x_prior, x_posterior, sigma_prior, sigma_posterior):
        self.Fusion(x_prior, x_posterior, sigma_prior, sigma_posterior)
        return self.m1x_posterior, self.m2x_posterior, self.Parameter, self.time_cost
