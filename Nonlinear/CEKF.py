import torch
from torch import autograd


class CEKF:
    def __init__(self, SystemModel, mode="full", archi="distributed"):
        self.sensor_num = SystemModel.sensor_num

        self.f = SystemModel.f
        self.m = SystemModel.m

        self.h_list = SystemModel.h_list
        self.n_list = SystemModel.n_list
        self.n = sum(self.n_list)

        self.R = SystemModel.R
        self.Q = SystemModel.Q

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        if mode == "full":
            self.fString = "ModAcc"
            self.hString = "ObsAcc"
            if archi == "centralized":
                self.hString = "ObsAcc_mul"
        elif mode == "partial":
            self.fString = "ModInacc"
            self.hString = "ObsInacc"

    def getJacobian(self, y, g):
        Jac = autograd.functional.jacobian(g, y)
        Jac = Jac.view(-1, self.m)
        return Jac

    def InitSequence(self, m2x_0, initial_state):
        self.m2x_0 = m2x_0
        self.m1x_posterior = initial_state
        self.m2x_posterior = m2x_0

    def Predict(self):
        self.m1x_prior = self.f(self.m1x_posterior)

        self.F = self.getJacobian(self.m1x_posterior, self.f)
        self.F_T = self.F.t()
        self.H_list = []
        for i in range(self.sensor_num):
            self.H_list.append(
                self.getJacobian(
                    self.m1x_posterior,
                    self.h_list[i],
                )
            )
        self.H = self.H_list[0]
        for i in range(1, self.sensor_num):
            self.H = torch.cat((self.H, self.H_list[i]), dim=0)
        self.H_T = self.H.t()

        self.m2x_prior = self.F @ self.m2x_posterior @ self.F_T + self.Q

        self.m1y = torch.zeros(self.n, 1)
        index_temp = 0
        for i in range(self.sensor_num):
            self.m1y[index_temp : index_temp + self.n_list[i]] = self.h_list[i](
                self.m1x_prior
            )
            index_temp += self.n_list[i]

        self.m2y = self.H @ self.m2x_prior @ self.H_T + self.R

    def KGain(self):
        self.KG = self.m2x_prior @ self.H_T @ torch.inverse(self.m2y)

    def Innovation(self, y):
        self.dy = y - self.m1y

    def Correct(self):
        self.m1x_posterior = self.m1x_prior + self.KG @ self.dy

        self.m2x_posterior = self.m2x_prior - self.KG @ self.m2y @ self.KG.t()

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()
        return self.m1x_prior, self.m1x_posterior, self.m2x_prior, self.m2x_posterior

    def GenerateSequence(self, y, T):
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])

        for t in range(0, T):
            yt = torch.unsqueeze(y[:, t], 1)
            _, xt, _, sigmat = self.Update(yt)
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)
