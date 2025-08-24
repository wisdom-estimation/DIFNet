"""# **Class: GRUNN**"""

import torch
import torch.nn as nn
import torch.nn.init as init

seed = 2023
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子


class NN(nn.Module):
    ###################
    ### Constructor ###
    ###################
    def __init__(self, ssModel, Batch_size):
        super().__init__()
        self.batch_size = Batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Build(ssModel)

    #############
    ### Build ###
    #############
    def Build(self, ssModel):

        self.m = ssModel.F.size()[0]

        self.n = ssModel.H.size()[0]

        self.sensor_num = ssModel.sensor_num

        # Number of neurons in the 1st net hidden layer
        H1_InfoNet = (ssModel.m**2 + ssModel.m**2) * 1 * 4 * self.sensor_num

        # Number of neurons in the 2nd net hidden layer
        H2_InfoNet = (ssModel.m**2) * 5 * 4

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
                m.weight.data.normal_(0, 0.01)
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
            self.device, non_blocking=True
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

        self.Parameter = L3_out_reshape

    ###############
    ### Forward ###
    ###############
    def forward(self, info_in):
        self.Estimation(info_in)
        return self.Parameter

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
