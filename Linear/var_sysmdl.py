import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


class SystemModel:

    def __init__(
        self,
        F,
        Q,
        H,
        H_mul,
        H_list,
        R,
        Trans,
        T,
        T_test,
        Sensor_num,
        prior_Q=None,
        prior_Sigma=None,
        prior_S=None,
    ):

        ####################
        ### Motion Model ###
        ####################
        self.F = F

        self.Q = Q
        self.m = self.Q.size()[0]

        #####################
        ### Sensor number ###
        #####################
        self.sensor_num = Sensor_num

        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.H_mul = H_mul
        self.H_list = H_list
        self.n_list = [H_list[i].size()[0] for i in range(self.sensor_num)]
        self.R = R
        self.n = self.R[0].size()[0]

        ########################################
        ### Internodal Transformation Matrix ###
        ########################################
        self.Trans = Trans
        self.Trans_F = []
        self.Trans_H = []
        self.Trans_Q = []
        self.Trans_R = []
        for i in range(Sensor_num):
            self.Trans_F.append(self.Trans[i] @ self.F @ torch.pinverse(self.Trans[i]))
            self.Trans_H.append(self.H_list[i] @ torch.pinverse(self.Trans[i]))
            self.Trans_Q.append(self.Trans[i] @ self.Q @ self.Trans[i].t())

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.eye(self.m)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0, initial_state_all):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0
        self.initial_state_all = initial_state_all

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T, sigma):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        R_gen_var = torch.empty_like(R_gen)
        Q_gen_var = torch.empty_like(Q_gen)

        m = Q_gen.size()[0]
        n = R_gen.size()[0]

        var = sigma

        # Generate Sequence Iteratively
        for t in range(0, T):
            R_gen_var = torch.zeros_like(R_gen)
            R_gen_var = (
                1 + var * torch.cos(torch.tensor(2 * (t / T) * torch.pi))
            ) * R_gen

            ########################
            #### State Evolution ###
            #######################
            xt = self.F @ self.x_prev

            # Process Noise
            mean = torch.zeros(self.m)
            distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
            eq = distrib.rsample()
            eq = torch.reshape(eq[:], [self.m, 1])

            # Additive Process Noise
            xt = xt.add(eq)

            ################
            ### Emission ###
            ################
            yt = self.H @ xt

            # Observation Noise
            mean = torch.zeros(self.n)
            distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen_var)
            er = distrib.rsample()
            er = torch.reshape(er[:], [self.n, 1])

            # Additive Observation Noise
            yt = yt.add(er)

            # yt = torch.zeros(size=[self.n, 1])
            # index_temp = 0
            # for i in range(self.sensor_num):
            #     yt[index_temp : index_temp + self.n_list[i], 0] = (
            #         self.H_list[i] @ xt
            #     ).squeeze()
            #     index_temp += self.n_list[i]

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, sigma, randomInit=False, seqInit=False, T_test=0):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):
            # Generate Sequence
            # print(i)
            # Randomize initial conditions to get a rich dataset
            if randomInit:
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if seqInit:
                initConditions = self.x_prev
                if (i * T % T_test) == 0:
                    initConditions = torch.zeros_like(self.m1x_0)

            self.InitSequence(initConditions, self.m2x_0, None)
            self.GenerateSequence(self.Q, self.R, T, sigma)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x

    def sampling(self, q, r, gain):

        if gain != 0:
            gain_q = 0.1
            # aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m)
            # aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m) + aq
        Q_gen = np.transpose(Aq) * Aq

        if gain != 0:
            gain_r = 0.5
            # ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n)
            # ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n) + ar
        R_gen = np.transpose(Ar) * Ar

        return [Q_gen, R_gen]
