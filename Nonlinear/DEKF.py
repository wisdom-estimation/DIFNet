import torch
from torch import autograd


class DEKF:
    def __init__(
        self,
        SystemModel,
        Trans,
        sensor_index,
        Batch_size,
        mode="full",
        archi="distributed",
    ):
        self.batch_size = Batch_size
        self.Trans = Trans
        self.sensor_index = sensor_index

        self.f = SystemModel.f
        self.m = self.Trans.size()[0]
        self.m_gobal = self.Trans.size()[1]

        self.h_list = SystemModel.h_list
        self.n = SystemModel.n

        self.R = SystemModel.R.repeat(self.batch_size, 1, 1)
        self.Q = (self.Trans @ SystemModel.Q @ self.Trans.t()).repeat(
            self.batch_size, 1, 1
        )

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
        Jac = Jac.view(-1, self.m_gobal)
        return Jac

    def InitSequence(self, m2x_0, initial_state):
        self.m2x_0 = m2x_0
        self.m2x_posterior = m2x_0
        self.m1x_posterior = initial_state

    def Predict(self):
        self.m1x_prior = torch.empty(size=[self.batch_size, self.m, 1])
        for i in range(self.batch_size):
            self.m1x_prior[i, :, :] = self.Trans @ self.f(
                torch.pinverse(self.Trans) @ self.m1x_posterior[i, :, :]
            )
        self.F = torch.empty(size=[self.batch_size, self.m, self.m])
        self.H = torch.empty(size=[self.batch_size, self.n, self.m])
        for i in range(self.batch_size):
            self.F[i, :, :] = (
                self.Trans
                @ self.getJacobian(
                    torch.pinverse(self.Trans) @ self.m1x_posterior[i, :, :],
                    self.f,
                )
                @ torch.pinverse(self.Trans)
            )
            self.H[i, :, :] = self.getJacobian(
                torch.pinverse(self.Trans) @ self.m1x_prior[i, :, :],
                self.h_list[self.sensor_index],
            ) @ torch.pinverse(self.Trans)
        self.F_T = self.F.transpose(1, 2)
        self.H_T = self.H.transpose(1, 2)

        self.m2x_prior = self.F @ self.m2x_posterior @ self.F_T + self.Q

        self.m1y = torch.empty(size=[self.batch_size, self.n, 1])
        for i in range(self.batch_size):
            self.m1y[i, :, :] = self.h_list[self.sensor_index](
                torch.pinverse(self.Trans) @ self.m1x_prior[i, :, :]
            )
        self.m2y = self.H @ self.m2x_prior @ self.H_T + self.R

    def KGain(self):
        self.KG = self.m2x_prior @ self.H_T @ torch.inverse(self.m2y)

    def Innovation(self, y):
        self.dy = y - self.m1y

    def Correct(self):
        self.m1x_posterior = self.m1x_prior + self.KG @ self.dy
        self.m2x_posterior = self.m2x_prior - self.KG @ self.m2y @ self.KG.transpose(
            1, 2
        )

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()
        return self.m1x_prior, self.m1x_posterior, self.m2x_prior, self.m2x_posterior
