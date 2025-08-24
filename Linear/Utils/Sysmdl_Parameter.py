import numpy as np
import torch
import math
from torch import autograd


class Sysmdl_Parameter:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        # traj len
        self.T = 50
        self.T_test = 50

        # sample size
        self.dt = 1

        # state dimension
        self.m = 6

        # observation dimension
        self.n = 6

        # Number of Training Examples
        self.N_E = 100

        # Number of Cross Validation Examples
        self.N_CV = 20

        # Number of Test Examples
        self.N_T = 40

        # Number of Sensor Num
        self.sensor_num = 5

        # Angle Velocity
        self.d_theta = torch.pi / 180

        # std variance
        self.q = 1
        self.filter_q = 5


class Model_s4(Sysmdl_Parameter):

    def __init__(self):
        super().__init__()
        self.F_cv = torch.eye(6)
        self.F_cv[0, 1] = self.dt
        self.F_cv[2, 3] = self.dt
        self.F_cv[4, 5] = self.dt

        self.sensor_num = 4

        self.H_cv1 = torch.cat((torch.eye(4), torch.zeros(4, 2)), dim=1)
        self.H_cv2 = self.H_cv1.clone()
        self.H_cv3 = torch.eye(6)
        self.H_cv4 = torch.cat((torch.zeros(2, 4), torch.eye(2)), dim=1)
        self.H_mul = torch.cat([self.H_cv1, self.H_cv2, self.H_cv3, self.H_cv4], dim=0)
        self.H_list = [self.H_cv1, self.H_cv2, self.H_cv3, self.H_cv4]
        self.n_list = [self.H_list[i].size()[0] for i in range(self.sensor_num)]

        self.Trans1 = self.H_cv1.clone()
        self.Trans2 = self.H_cv2.clone()
        self.Trans3 = self.H_cv3.clone()
        self.Trans4 = self.H_cv4.clone()
        self.Trans = [self.Trans1, self.Trans2, self.Trans3, self.Trans4]

        self.Q_cv = (self.q**2) * torch.kron(
            torch.eye(3),
            torch.tensor(
                [[self.dt**4 / 3, self.dt**3 / 2], [self.dt**3 / 2, self.dt**2]],
                dtype=torch.float32,
            ),
        )
        self.r1 = torch.tensor([100, 10, 100, 10], dtype=torch.float32)
        self.r2 = torch.tensor([200, 20, 200, 20], dtype=torch.float32)
        # self.r2 = torch.tensor([100, 10, 100, 10], dtype=torch.float32)
        self.r3 = torch.tensor([200, 20, 200, 20, 200, 20], dtype=torch.float32)
        # self.r3 = torch.tensor([100, 10, 100, 10, 100, 10], dtype=torch.float32)
        self.r4 = torch.tensor([100, 10], dtype=torch.float32)

        # self.r1 = self.r1 * 2
        # self.r2 = self.r2 * 2
        # self.r3 = self.r3 * 2
        # self.r4 = self.r4 * 2

        self.R_list = [
            torch.diag(self.r1**2),
            torch.diag(self.r2**2),
            torch.diag(self.r3**2),
            torch.diag(self.r4**2),
        ]

        beta = [0.5 for _ in range(self.sensor_num)]
        R_0 = torch.diag(torch.tensor([100.0, 10.0, 100.0, 10.0, 100.0, 10.0]) ** 2)
        self.R_mul = torch.zeros(size=[sum(self.n_list), sum(self.n_list)])
        index_1 = 0
        for i in range(self.sensor_num):
            index_2 = 0
            for j in range(self.sensor_num):
                self.R_mul[
                    index_1 : index_1 + self.n_list[i],
                    index_2 : index_2 + self.n_list[j],
                ] = (
                    beta[i] * beta[j] * self.Trans[i] @ R_0 @ self.Trans[j].t()
                )
                if i == j:
                    self.R_mul[
                        index_1 : index_1 + self.n_list[i],
                        index_2 : index_2 + self.n_list[j],
                    ] += self.R_list[i]
                index_2 += self.n_list[j]
            index_1 += self.n_list[i]

        self.x0 = torch.tensor(
            [[0], [100], [0], [100], [0], [100]], dtype=torch.float32
        )
        self.P0 = 10000 * torch.eye(6)

        self.R_list = []
        index_temp = 0
        for i in range(self.sensor_num):
            self.R_list.append(
                self.R_mul[
                    index_temp : index_temp + self.n_list[i],
                    index_temp : index_temp + self.n_list[i],
                ]
            )
            index_temp += self.n_list[i]

        self.filter_F_cv = self.F_cv
        self.filter_H_list = self.H_list
        self.filter_Q_cv = self.filter_q**2 * torch.kron(
            torch.eye(3),
            torch.tensor(
                [[self.dt**4 / 3, self.dt**3 / 2], [self.dt**3 / 2, self.dt**2]],
                dtype=torch.float32,
            ),
        )
        self.filter_R_mul = self.R_mul
        self.filter_R_list = self.R_list
        self.filter_x0 = self.x0
        self.filter_P0 = self.P0

        self.filter_inexactIF_Q_cv = self.filter_Q_cv
        self.filter_inexact_R_cv = torch.diag(
            torch.cat(
                [
                    (self.r1 / 1) ** 1,
                    (self.r2 / 1) ** 1,
                    (self.r3 / 1) ** 1,
                    (self.r4 / 1) ** 1,
                ],
                dim=0,
            )
        )
        self.inexact_R_list = []
        index_temp = 0
        for i in range(self.sensor_num):
            self.inexact_R_list.append(
                self.filter_inexact_R_cv[
                    index_temp : index_temp + self.n_list[i],
                    index_temp : index_temp + self.n_list[i],
                ]
            )
            index_temp += self.n_list[i]


if __name__ == "__main__":
    model = Model_s4()
    print(model.R_mul)
    print(model.filter_inexact_R_cv)
