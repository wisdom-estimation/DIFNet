import torch
from DEKF import DEKF
from torch import autograd
import time

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")
# dev = torch.device("cpu")
# print("Running on the CPU")


class Extended_DIF:
    def __init__(self, SystemModel, SysModel_EKF, mode="full"):
        self.sensor_num = SystemModel.sensor_num
        self.Trans = SystemModel.Trans

        # Set State Evolution Function
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.Q = SystemModel.Q

        # Set Observation Function
        self.h_list = SystemModel.h_list
        self.n_list = SystemModel.n_list
        self.R_mul = SystemModel.R
        self.R_mul_Inverse = torch.inverse(self.R_mul)

        # Set Local Filters
        self.Filters = []
        for i in range(self.sensor_num):
            sys = SysModel_EKF[i]
            sys.n = self.n_list[i]
            self.Filters.append(DEKF(sys, self.Trans[i], i, 1))

        # Set the length of Traj
        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Set Jacobian function
        # self.getJacobian = SystemModel.getJacobian

        # Full knowledge about the model or partial? (Should be made more elegant)
        if mode == "full":
            self.fString = "ModAcc"
            self.hString = "ObsAcc"
            if SystemModel.sensor_num > 1:
                self.hString = "ObsAcc_mul"
        elif mode == "partial":
            self.fString = "ModInacc"
            self.hString = "ObsInacc"

    def getJacobian(self, y, g):
        Jac = autograd.functional.jacobian(g, y)
        Jac = Jac.view(-1, self.m)
        return Jac

    def InitSequence(self, m1x_0, m2x_0, initial_state):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0
        self.m1x_posterior = [
            self.Trans[i] @ initial_state for i in range(self.sensor_num)
        ]
        self.m2x_posterior = [
            self.Trans[i] @ m2x_0 @ self.Trans[i].t() for i in range(self.sensor_num)
        ]

    def Predict(self):
        self.info = []
        self.Info = []
        self.F_list = []
        self.H_list = []
        for i in range(self.sensor_num):
            self.m1x_prior = self.Trans[i] @ self.f(
                torch.pinverse(self.Trans[i]) @ self.m1x_posterior[i]
            )
            self.F_list.append(
                self.Trans[i]
                @ self.getJacobian(
                    torch.pinverse(self.Trans[i]) @ self.m1x_posterior[i], self.f
                )
                @ torch.pinverse(self.Trans[i])
            )
            self.H_list.append(
                self.getJacobian(
                    torch.pinverse(self.Trans[i]) @ self.m1x_posterior[i],
                    self.h_list[i],
                )
                @ torch.pinverse(self.Trans[i])
            )
            self.m2x_prior = (
                self.F_list[i] @ self.m2x_posterior[i] @ self.F_list[i].t()
                + self.Trans[i] @ self.Q @ self.Trans[i].t()
            )
            self.Info.append(torch.inverse(self.m2x_prior))
            self.info.append(self.Info[i] @ self.m1x_prior)
        self.H_mul = self.H_list[0] @ self.Trans[0]
        for i in range(1, self.sensor_num):
            self.H_mul = torch.cat((self.H_mul, self.H_list[i] @ self.Trans[i]), dim=0)

    def Update(self, x_prior, x_posterior, sigma_prior, sigma_posterior):
        self.time_cost = time.perf_counter()
        for i in range(self.sensor_num):
            info = torch.zeros(self.Trans[i].shape[0], 1)
            Info = torch.zeros(self.Trans[i].shape[0], self.Trans[i].shape[0])
            index_temp = 0
            for j in range(self.sensor_num):
                M = (
                    (
                        torch.pinverse(self.Trans[i]).t()
                        @ self.H_mul.t()
                        @ self.R_mul_Inverse[
                            :, index_temp : index_temp + self.n_list[j]
                        ]
                        @ self.Filters[j].R
                        @ torch.pinverse(self.H_list[j] @ self.Trans[j]).transpose(0, 1)
                        @ self.Trans[i].t()
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                index_temp += self.n_list[j]
                info += (
                    M
                    @ (self.Trans[j] @ torch.pinverse(self.Trans[i])).T
                    @ (
                        torch.inverse(sigma_posterior[j]) @ x_posterior[j]
                        - torch.inverse(sigma_prior[j]) @ x_prior[j]
                    )
                )
                Info += (
                    M
                    @ (self.Trans[j] @ torch.pinverse(self.Trans[i])).T
                    @ (
                        torch.inverse(sigma_posterior[j])
                        - torch.inverse(sigma_prior[j])
                    )
                    @ self.Trans[j]
                    @ torch.pinverse(self.Trans[i])
                )
            self.m2x_posterior[i] = torch.inverse(self.Info[i] + Info)
            self.m1x_posterior[i] = self.m2x_posterior[i] @ (self.info[i] + info)
        self.time_cost = time.perf_counter() - self.time_cost

    def Fusion(self, x_prior, x_posterior, sigma_prior, sigma_posterior):
        self.Predict()
        self.Update(x_prior, x_posterior, sigma_prior, sigma_posterior)
        return self.m1x_posterior, self.m2x_posterior, self.time_cost

    def GenerateSequence(self, y, T):
        self.x = [torch.zeros(self.m, T).to(dev) for _ in range(self.sensor_num)]
        self.sigma = [
            torch.zeros(size=[self.m, self.m, T]).to(dev)
            for _ in range(self.sensor_num)
        ]
        self.time_cost_total = 0
        [
            self.Filters[i].InitSequence(
                self.m2x_posterior[i].repeat(1, 1, 1),
                self.m1x_posterior[i].repeat(1, 1, 1),
            )
            for i in range(self.sensor_num)
        ]
        for t in range(0, T):
            x_prior = []
            x_posterior = []
            sigma_prior = []
            sigma_posterior = []

            if self.sensor_num == 1:
                yt = torch.unsqueeze(y[:, t], 1)
                (
                    x_prior[:, :],
                    x_posterior[:, :],
                    sigma_prior[:, :],
                    sigma_posterior[:, :],
                ) = self.Filters[0].Update(yt)
                xt, sigmat = self.Fusion(
                    x_prior, x_posterior, sigma_prior, sigma_posterior
                )
            else:
                index_temp = 0
                for i in range(self.sensor_num):
                    yt = torch.unsqueeze(
                        y[index_temp : index_temp + self.n_list[i], t], 1
                    )
                    index_temp += self.n_list[i]
                    x_prior_i, x_posterior_i, sigma_prior_i, sigma_posterior_i = (
                        self.Filters[i].Update(yt)
                    )
                    x_prior.append(x_prior_i.squeeze(0))
                    x_posterior.append(x_posterior_i.squeeze(0))
                    sigma_prior.append(sigma_prior_i.squeeze(0))
                    sigma_posterior.append(sigma_posterior_i.squeeze(0))
                xt, sigmat, time_t = self.Fusion(
                    x_prior, x_posterior, sigma_prior, sigma_posterior
                )
                self.time_cost_total += time_t
            for i in range(self.sensor_num):
                self.x[i][:, t] = torch.squeeze(torch.pinverse(self.Trans[i]) @ xt[i])
                self.sigma[i][:, :, t] = torch.squeeze(
                    torch.pinverse(self.Trans[i])
                    @ sigmat[i]
                    @ torch.pinverse(self.Trans[i]).t()
                )
