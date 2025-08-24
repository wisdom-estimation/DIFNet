"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""

import torch
import time
from KF import KalmanFilter

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


class Dencentralized_InformationFilter:

    def __init__(self, SystemModel, SysModel_KF, sensor_num):
        self.SystemModel = SystemModel
        self.Trans = SystemModel.Trans
        self.F = SystemModel.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.Trans_F = SystemModel.Trans_F
        self.m = SystemModel.m

        self.Q = SystemModel.Q
        self.Trans_Q = SystemModel.Trans_Q

        self.H = SystemModel.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.H_mul = SystemModel.H_mul
        self.H_list = SystemModel.H_list
        self.Trans_H = SystemModel.Trans_H
        self.n = SystemModel.n
        self.n_list = SystemModel.n_list

        self.R = SystemModel.R
        self.R_Inverse = torch.inverse(self.R)

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        self.sensor_num = sensor_num

        self.Filters = []
        for i in range(self.sensor_num):
            sys = SysModel_KF[i]
            sys.F = self.Trans_F[i]
            sys.H = self.Trans_H[i]
            sys.Q = self.Trans_Q[i]
            self.Filters.append(KalmanFilter(sys, 1))

    def InitSequence(self, m1x_0, m2x_0, initial_state):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0
        self.m1x_posterior = [
            self.Trans[i] @ initial_state for i in range(self.sensor_num)
        ]
        self.m2x_posterior = [
            self.Trans[i] @ m2x_0 @ self.Trans[i].t() for i in range(self.sensor_num)
        ]

    # Predict
    def Predict(self):
        self.info = []
        self.Info = []
        for i in range(self.sensor_num):
            # Predict the 1-st moment of x
            self.m1x_prior = self.Trans_F[i] @ self.m1x_posterior[i]

            # Predict the 2-nd moment of x
            self.m2x_prior = torch.matmul(self.Trans_F[i], self.m2x_posterior[i])
            self.m2x_prior = (
                torch.matmul(self.m2x_prior, self.Trans_F[i].t())
                + self.Trans[i] @ self.Q @ self.Trans[i].t()
            )

            # Transform the 2-nd moment of x into information matrix
            self.Info.append(torch.inverse(self.m2x_prior))

            # Transform the 1-st moment of x into information vector
            self.info.append(self.Info[i] @ self.m1x_prior)

    # Innovation
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
                        @ self.R_Inverse[:, index_temp : index_temp + self.n_list[j]]
                        @ self.Filters[j].R
                        @ torch.pinverse(
                            self.Filters[j].H.squeeze(0) @ self.Trans[j]
                        ).transpose(0, 1)
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

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = [torch.zeros(self.m, T).to(dev) for i in range(self.sensor_num)]
        self.sigma = [
            torch.zeros(size=[self.m, self.m, T]).to(dev)
            for i in range(self.sensor_num)
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
