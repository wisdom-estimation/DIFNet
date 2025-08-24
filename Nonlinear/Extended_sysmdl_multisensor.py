import torch
from torch.distributions.multivariate_normal import MultivariateNormal

variance = 1


class SystemModel:

    def __init__(
        self,
        f,
        Q,
        h,
        h_list,
        n_list,
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
        self.f = f

        self.Q = Q
        self.m = self.Q.size()[0]

        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.h_list = h_list

        self.n_list = n_list

        self.R = R
        self.n = self.R[0].size()[0]

        self.Trans = Trans

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #####################
        ### Sensor number ###
        #####################
        self.sensor_num = Sensor_num

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
        self.m2x_0 = m2x_0
        if initial_state_all:
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
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        # Generate Sequence Iteratively
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################
            xt = self.f(self.x_prev)
            # Process Noise
            mean = torch.zeros([self.m])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
            eq = distrib.rsample()
            eq = torch.reshape(eq[:], [self.m, 1])
            # Additive Process Noise
            xt = torch.add(xt, eq)

            ################
            ### Emission ###
            ################
            # yt = self.h(xt)

            yt = torch.zeros(size=[self.n, 1])
            index_temp = 0
            for i in range(self.sensor_num):
                yt[index_temp : index_temp + self.n_list[i], 0] = self.h_list[i](
                    xt
                ).squeeze()
                index_temp += self.n_list[i]

            # Observation Noise
            mean = torch.zeros([self.n])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
            er = distrib.rsample()
            er = torch.reshape(er[:], [self.n, 1])

            # Additive Observation Noise
            yt = torch.add(yt, er)

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
    def GenerateBatch(self, size, T, randomInit=False):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        initConditions = self.m1x_0

        ### Generate Examples
        for i in range(0, size):
            # Generate Sequence
            print(i)
            # Randomize initial conditions to get a rich dataset
            if randomInit:
                initConditions = torch.rand_like(self.m1x_0) * variance
            self.InitSequence(initConditions, self.m2x_0, self.initial_state_all)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x
