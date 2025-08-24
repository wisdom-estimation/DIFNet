"""# **Class: Extended Kalman Filter**
Theoretical NonLinear Kalman Filter
"""

import time
import torch

if torch.cuda.is_available():
    cuda0 = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")


# cuda0 = torch.device("cpu")
# print("Running on the CPU")
class ExtendedKalmanFilter:
    def __init__(self, SystemModel, Batch_size=20, mode="full", archi="distributed"):
        sensor_num = SystemModel.sensor_num
        self.f = SystemModel.f
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q.repeat(Batch_size, sensor_num, 1, 1)

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R.repeat(Batch_size, sensor_num, 1, 1)

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        self.batch_size = Batch_size
        self.sensor_num = SystemModel.sensor_num

        # Set Jacobian function
        self.getJacobian = SystemModel.getJacobian

        # Full knowledge about the model or partial? (Should be made more elegant)
        if mode == "full":
            self.fString = "ModAcc"
            self.hString = "ObsAcc"
            if archi == "centralized":
                self.hString = "ObsAcc_mul"
        elif mode == "partial":
            self.fString = "ModInacc"
            self.hString = "ObsInacc"

    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.zeros_like(self.m1x_posterior)
        for i in range(0, self.m1x_posterior.size()[0]):
            for j in range(0, self.m1x_posterior.size()[1]):
                self.m1x_prior[i, j, :, :] = self.f(self.m1x_posterior[i, j, :, :])

        # Compute the Jacobians
        self.F = torch.zeros(
            size=[
                self.m1x_posterior.size()[0],
                self.m1x_posterior.size()[1],
                self.m,
                self.m,
            ]
        )
        self.H = torch.zeros(
            size=[
                self.m1x_posterior.size()[0],
                self.m1x_posterior.size()[1],
                self.n,
                self.m,
            ]
        )
        # self.UpdateJacobians(
        #     self.getJacobian(self.m1x_posterior, self.fString),
        #     self.getJacobian(self.m1x_prior, self.hString),
        # )
        for i in range(self.m1x_posterior.size()[0]):
            for j in range(self.m1x_posterior.size()[1]):
                self.F[i, j, :, :] = self.getJacobian(
                    self.m1x_posterior[i, j, :, :], self.fString
                )
                self.H[i, j, :, :] = self.getJacobian(
                    self.m1x_prior[i, j, :, :], self.hString
                )
        self.F_T = torch.transpose(self.F, 2, 3)
        self.H_T = torch.transpose(self.H, 2, 3)
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.zeros(size=[self.batch_size, self.sensor_num, self.n, 1])
        for i in range(0, self.m1x_prior.size()[0]):
            for j in range(0, self.m1x_prior.size()[1]):
                self.m1y[i, j, :, :] = self.h(self.m1x_prior[i, j, :, :])
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

        # #Save KalmanGain
        # self.KG_array[self.i] = self.KG
        # self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 2, 3))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()
        return self.m1x_prior, self.m1x_posterior, self.m2x_prior, self.m2x_posterior

    def InitSequence(self, m2x_0, initial_state):
        self.m2x_0 = m2x_0
        self.m1x_posterior = initial_state
        self.m2x_posterior = m2x_0
        # 无batch，有sensor_num
        if len(self.m1x_posterior.shape) == 3:
            self.m1x_posterior = self.m1x_posterior.unsqueeze(0)
            self.m2x_posterior = self.m2x_posterior.unsqueeze(0)
        # 无batch，无sensor_num
        if len(self.m1x_posterior.shape) == 2:
            self.m1x_posterior = self.m1x_posterior.unsqueeze(0).unsqueeze(0)
            self.m2x_posterior = self.m2x_posterior.unsqueeze(0).unsqueeze(0)
        # Fusion time
        self.fusion_time = 0

    # Update Jacobians F and H
    # def UpdateJacobians(self, F, H):
    #     F_temp = F.repeat(self.batch_size, self.sensor_num, 1, 1)
    #     H_temp = H.repeat(self.batch_size, self.sensor_num, 1, 1)
    #     return F_temp, H_temp
    # print(self.H,self.F,'\n')

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        # # Pre allocate KG array
        # self.KG_array = torch.zeros((T,self.m,self.n))
        # self.i = 0 # Index for KG_array alocation

        for t in range(0, T):
            yt = torch.unsqueeze(y[:, t], 1)
            tic = time.time()
            _, xt, _, sigmat = self.Update(yt)
            toc = time.time()
            self.fusion_time += toc - tic
            self.x[:, t] = torch.squeeze(xt)[0]
            self.sigma[:, :, t] = torch.squeeze(sigmat)[0]
