import numpy as np
import torch


class Sysmdl_Parameter:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


class Model_3DCT(Sysmdl_Parameter):

    def __init__(self):
        super().__init__()
        self.sensor_num = 4
        self.m = 6
        omega = torch.tensor([0.05], dtype=torch.float32)
        dt = torch.tensor([1], dtype=torch.float32)
        self.F = torch.tensor(
            [
                [
                    1,
                    torch.sin(omega * dt) / omega,
                    0,
                    -(1 - torch.cos(omega * dt)) / omega,
                    0,
                    0,
                ],
                [0, torch.cos(omega * dt), 0, -torch.sin(omega * dt), 0, 0],
                [
                    0,
                    (1 - torch.cos(omega * dt)) / omega,
                    1,
                    torch.sin(omega * dt) / omega,
                    0,
                    0,
                ],
                [0, torch.sin(omega * dt), 0, torch.cos(omega * dt), 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        self.Q = self.q**2 * torch.tensor(
            [
                [
                    2 * (omega * dt - torch.sin(omega * dt)) / omega**3,
                    (1 - torch.cos(omega * dt)) / omega**2,
                    0.0,
                    (omega * dt - torch.sin(omega * dt)) / omega**2,
                    0,
                    0,
                ],
                [
                    (1 - torch.cos(omega * dt)) / omega**2,
                    dt,
                    -(omega * dt - torch.sin(omega * dt)) / omega**2,
                    0.0,
                    0,
                    0,
                ],
                [
                    0.0,
                    -(omega * dt - torch.sin(omega * dt)) / omega**2,
                    2 * (omega * dt - torch.sin(omega * dt)) / omega**3,
                    (1 - torch.cos(omega * dt)) / omega**2,
                    0,
                    0,
                ],
                [
                    (omega * dt - torch.sin(omega * dt)) / omega**2,
                    0.0,
                    (1 - torch.cos(omega * dt)) / omega**2,
                    dt,
                    0,
                    0,
                ],
                [0, 0, 0, 0, dt**4 / 3, dt**3 / 2],
                [0, 0, 0, 0, dt**3 / 2, dt**2],
            ],
            dtype=torch.float32,
        )

        self.sensor1 = torch.tensor([-5500, 1000, 0], dtype=torch.float32)
        self.sensor2 = torch.tensor([-5000, 0, 0], dtype=torch.float32)
        self.sensor3 = torch.tensor([500, 300, 0], dtype=torch.float32)
        self.sensor4 = torch.tensor([50, 500, 0], dtype=torch.float32)
        self.sensor_list = [self.sensor1, self.sensor2, self.sensor3, self.sensor4]

        self.Trans1 = torch.cat([torch.eye(4), torch.zeros(4, 2)], dim=1)
        self.Trans2 = torch.cat([torch.eye(4), torch.zeros(4, 2)], dim=1)
        self.Trans3 = torch.eye(6)
        self.Trans4 = torch.cat([torch.zeros(2, 4), torch.eye(2)], dim=1)
        self.Trans = [self.Trans1, self.Trans2, self.Trans3, self.Trans4]

        self.r1 = torch.tensor([1 * torch.pi / 180, 15], dtype=torch.float32)
        self.r2 = torch.tensor([250, 25], dtype=torch.float32)
        self.r3 = torch.tensor([200, 20, 200, 20, 200, 20], dtype=torch.float32)
        self.r4 = torch.tensor([100, 10], dtype=torch.float32)

        self.n_list = [2, 2, 6, 2]
        self.x0 = torch.tensor(
            [[0], [100], [0], [100], [0], [100]], dtype=torch.float32
        )
        self.P0 = 10000 * torch.eye(6)

        self.R_list = [
            torch.diag(self.r1**2),
            torch.diag(self.r2**2),
            torch.diag(self.r3**2),
            torch.diag(self.r4**2),
        ]

        beta = [2 for _ in range(self.sensor_num)]
        R_0 = torch.tensor(
            [1 * torch.pi / 180, 150, 15, 100, 10, 100, 10, 100, 10],
            dtype=torch.float32,
        )
        R_0 = torch.diag(R_0**2)

        self.T1 = torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        self.T2 = torch.tensor(
            [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        self.T3 = torch.cat([torch.zeros(6, 3), torch.eye(6)], dim=1)
        self.T4 = torch.cat([torch.zeros(2, 7), torch.eye(2)], dim=1)
        self.T_list = [self.T1, self.T2, self.T3, self.T4]
        self.R_mul = torch.zeros(size=[sum(self.n_list), sum(self.n_list)])
        index_1 = 0
        for i in range(self.sensor_num):
            index_2 = 0
            for j in range(self.sensor_num):
                self.R_mul[
                    index_1 : index_1 + self.n_list[i],
                    index_2 : index_2 + self.n_list[j],
                ] = (
                    beta[i] * beta[j] * self.T_list[i] @ R_0 @ self.T_list[j].t()
                )
                if i == j:
                    self.R_mul[
                        index_1 : index_1 + self.n_list[i],
                        index_2 : index_2 + self.n_list[j],
                    ] += self.R_list[i]
                index_2 += self.n_list[j]
            index_1 += self.n_list[i]

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

        self.inexact_Q = self.Q / self.q**2 * self.filter_q**2
        self.inexact_R = torch.diag(
            torch.cat(
                [
                    self.r1 / 1**1,
                    self.r2 / 1**1,
                    self.r3 / 1**1,
                    self.r4 / 1**1,
                ],
                dim=0,
            )
        )
        self.inexact_R_list = []
        index_temp = 0
        for i in range(self.sensor_num):
            self.inexact_R_list.append(
                self.inexact_R[
                    index_temp : index_temp + self.n_list[i],
                    index_temp : index_temp + self.n_list[i],
                ]
            )
            index_temp += self.n_list[i]

        self.h_list = [self.h1, self.h2, self.h3, self.h4]

    def f(self, x):
        return self.F @ x

    def h1(self, x):
        y = torch.zeros(size=[2, 1])
        y[0] = torch.atan2(x[2] - self.sensor_list[0][1], x[0] - self.sensor_list[0][0])
        y[1] = torch.sqrt(x[3] ** 2 + x[1] ** 2)
        return y

    def h2(self, x):
        y = torch.zeros(size=[2, 1])
        y[0] = torch.sqrt(
            (x[2] - self.sensor_list[1][1]) ** 2 + (x[0] - self.sensor_list[1][0]) ** 2
        )
        y[1] = torch.sqrt(x[3] ** 2 + x[1] ** 2)
        return y

    def h3(self, x):
        return torch.eye(self.m) @ x

    def h4(self, x):
        return self.Trans4 @ x
