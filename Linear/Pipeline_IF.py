import torch
import torch.nn as nn
import os
import random
from Utils.Plot import Plot
import matplotlib.pyplot as plt
from KF import KalmanFilter
from tqdm import tqdm
import time
from itertools import chain
import numpy as np
import time

# plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# plt.rcParams["figure.frameon"] = False
# plt.rcParams["figure.constrained_layout.use"] = True
# plt.rcParams["axes.unicode_minus"] = False

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")


class Pipeline_IF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + "/"
        self.modelName = modelName
        self.modelFileName = (
            self.folderName + "model/" + "model_" + self.modelName + ".pt"
        )
        self.PipelineName = (
            self.folderName + "pipeline/" + "pipeline_" + self.modelName + ".pt"
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save(self):
        torch.save(self, self.PipelineName)

    def save_checkpoint(self, state, filename="checkpoint.pth.tar"):
        torch.save(state, filename)

    def setssModel(self, ssModel, ssModel_kf, Batch_size):
        self.N_B = Batch_size
        self.ssModel = ssModel
        self.sensor_num = ssModel.sensor_num
        self.n_list = ssModel.n_list
        self.Filters = []
        # self.DIFNet = DIFNet(ssModel, ssModel_kf, ssModel.sensor_num, self.N_B)
        # self.model = self.DIFNet.NN.to(self.device)
        for i in range(ssModel.sensor_num):
            sys = ssModel_kf[i]
            sys.F = ssModel.Trans_F[i]
            sys.H = ssModel.Trans_H[i]
            sys.Q = ssModel.Trans_Q[i]
            self.Filters.append(KalmanFilter(sys, self.N_B))

    def setModel(self, model):
        self.model = model.to(self.device)
        self.model_list = [nn.to(self.device) for nn in model.NN_list]
        self.parameters = chain(*[m.parameters() for m in self.model_list])

    def load_Model(self, filename):
        self.model = torch.load(filename).to(self.device)

    def Plot(self):
        self.Plot = Plot(self.folderName, self.modelName)

    def setTrainingParams(self, n_Epochs, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        # self.N_B = n_Batch  # Number of Samples in Batch
        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay
        self.TrainLossFileName = (
            self.folderName + "loss/" + self.modelName + "_MSE_train_cv_loss" + ".pt"
        )
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction="mean")

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(
            self.parameters, lr=self.learningRate, weight_decay=self.weightDecay
        )
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=1e-3,
            max_lr=1e-3,
            step_size_up=20,
            step_size_down=20,
            cycle_momentum=False,
        )

    def NNTest(self, n_Test, test_loader, position_idx, velocity_idx):
        self.N_T = n_Test
        self.MSE_test_linear_arr = torch.zeros(size=[self.N_T])
        loss_fn = nn.MSELoss(reduction="mean")
        self.model = torch.load(self.modelFileName, map_location=dev)

        self.model.eval()
        for sen in range(self.sensor_num):
            self.model.NN_list[sen].eval()

        with torch.no_grad():

            x_out_test_total = [
                torch.zeros(self.N_T, self.ssModel.m, self.ssModel.T)
                for _ in range(self.sensor_num)
            ]

            x_out_test = [
                torch.zeros(self.N_B, self.ssModel.m, self.ssModel.T)
                for _ in range(self.sensor_num)
            ]

            self.MSE_test_linear_seq = [
                torch.zeros(size=[self.ssModel.T]) for i in range(self.sensor_num)
            ]
            self.MSE_test_linear_seq_pos = [
                torch.zeros(size=[self.ssModel.T]) for i in range(self.sensor_num)
            ]
            self.MSE_test_linear_seq_vel = [
                torch.zeros(size=[self.ssModel.T]) for i in range(self.sensor_num)
            ]
            time_cost_total = 0
            for index, [test_input, test_target, test_initial_state_all] in tqdm(
                enumerate(test_loader)
            ):
                for sen in range(self.sensor_num):
                    self.model.NN_list[sen].init_hidden()
                self.model.InitSequence(
                    self.ssModel.m1x_0.repeat(self.N_B, 1, 1),
                    self.ssModel.m2x_0.repeat(self.N_B, 1, 1),
                    test_initial_state_all,
                )
                [
                    self.Filters[i].InitSequence(
                        self.ssModel.Trans[i].repeat(self.N_B, 1, 1)
                        @ self.ssModel.m2x_0
                        @ self.ssModel.Trans[i].repeat(self.N_B, 1, 1).transpose(1, 2),
                        self.ssModel.Trans[i].repeat(self.N_B, 1, 1)
                        @ self.ssModel.m1x_0,
                    )
                    for i in range(self.ssModel.sensor_num)
                ]
                for t in range(0, self.ssModel.T):
                    x_prior = []
                    x_posterior = []
                    sigma_prior = []
                    sigma_posterior = []
                    index_temp = 0
                    for i in range(self.sensor_num):
                        yt = torch.unsqueeze(
                            test_input[:, index_temp : index_temp + self.n_list[i], t],
                            -1,
                        )
                        index_temp += self.n_list[i]
                        x_prior_i, x_posterior_i, sigma_prior_i, sigma_posterior_i = (
                            self.Filters[i].Update(yt)
                        )
                        x_prior.append(x_prior_i)
                        x_posterior.append(x_posterior_i)
                        sigma_prior.append(sigma_prior_i)
                        sigma_posterior.append(sigma_posterior_i)
                    xt, _, _, time_t = self.model(
                        x_prior, x_posterior, sigma_prior, sigma_posterior
                    )
                    time_cost_total += time_t
                    for i in range(self.sensor_num):
                        x_out_test[i][:, :, t] = xt[i].squeeze(-1)
                        x_out_test_total[i][
                            index * self.N_B : (index + 1) * self.N_B, :, t
                        ] = x_out_test[i][:, :, t]
                for i in range(self.sensor_num):
                    Loss = self.loss_fn(
                        self.ssModel.Trans[i].repeat(self.N_B, 1, 1) @ x_out_test[i],
                        self.ssModel.Trans[i].repeat(self.N_B, 1, 1)
                        @ test_target[:, :, :],
                    )
                    self.MSE_test_linear_arr[index] += Loss.item()

                    adjusted_x_out = x_out_test[i].permute(1, 0, 2)
                    adjusted_target = test_target.permute(1, 0, 2)
                    for j in range(self.ssModel.T):
                        err = (
                            self.ssModel.Trans[i] @ adjusted_x_out[:, :, j]
                            - self.ssModel.Trans[i] @ adjusted_target[:, :, j]
                        ) ** 2
                        err_pos = (
                            (self.ssModel.Trans[i] @ adjusted_x_out[:, :, j])[
                                position_idx[i], :
                            ]
                            - (self.ssModel.Trans[i] @ adjusted_target[:, :, j])[
                                position_idx[i], :
                            ]
                        ) ** 2
                        err_vel = (
                            (self.ssModel.Trans[i] @ adjusted_x_out[:, :, j])[
                                velocity_idx[i], :
                            ]
                            - (self.ssModel.Trans[i] @ adjusted_target[:, :, j])[
                                velocity_idx[i], :
                            ]
                        ) ** 2
                        self.MSE_test_linear_seq[i][j] += torch.mean(err) * self.N_B
                        self.MSE_test_linear_seq_pos[i][j] += (
                            torch.mean(err_pos) * self.N_B
                        )
                        self.MSE_test_linear_seq_vel[i][j] += (
                            torch.mean(err_vel) * self.N_B
                        )
            for i in range(self.sensor_num):
                self.MSE_test_linear_seq[i] = torch.sqrt(
                    self.MSE_test_linear_seq[i] / self.N_T
                )
                self.MSE_test_linear_seq_pos[i] = torch.sqrt(
                    self.MSE_test_linear_seq_pos[i] / self.N_T
                )
                self.MSE_test_linear_seq_vel[i] = torch.sqrt(
                    self.MSE_test_linear_seq_vel[i] / self.N_T
                )
            # Average
            self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
            self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

            # Print MSE Cross Validation
            str = self.modelName + "-" + "MSE Test:"
            print(str, self.MSE_test_dB_avg, "[dB]")
            return (
                self.MSE_test_linear_arr,
                self.MSE_test_linear_avg,
                self.MSE_test_dB_avg,
                x_out_test_total,
                self.MSE_test_linear_seq,
                self.MSE_test_linear_seq_pos,
                self.MSE_test_linear_seq_vel,
                time_cost_total,
            )

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(
            self.N_Epochs,
            MSE_KF_dB_avg,
            self.MSE_test_dB_avg,
            self.MSE_cv_dB_epoch,
            self.MSE_train_dB_epoch,
        )

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

    def Test_mse_seq(self, test_numpy, targets_numpy, state_idx):
        MSE = torch.zeros((test_numpy.size()[2], 1))
        for i in range(test_numpy.size()[2]):
            MSE[i] = torch.sqrt(
                torch.mean(
                    (test_numpy[:, state_idx[0], i] - targets_numpy[:, state_idx[0], i])
                    ** 2
                    + (
                        test_numpy[:, state_idx[1], i]
                        - targets_numpy[:, state_idx[1], i]
                    )
                    ** 2
                )
            )
        return MSE

    def PlotTrain_MSE(self, MSE_train, MSE_cv, MSE_train_db=None, MSE_cv_db=None):
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(MSE_train)), MSE_train, "b", label="MSE_train")
        plt.plot(range(len(MSE_train)), MSE_cv, "r", label="MSE_cv")
        plt.legend()
        plt.xlabel("Training    epoch")
        plt.ylabel("MSE    Loss")
        plt.show()

    def PlotTrain_MSE_compare(self, MSE_train_InfomationNet, MSE_train_KalmanNet):
        plt.figure(figsize=(15, 10))
        plt.plot(
            range(len(MSE_train_InfomationNet)),
            MSE_train_InfomationNet,
            "-",
            color="r",
            markersize=10,
            label="MSE_train_IFNet",
        )
        plt.plot(
            range(len(MSE_train_KalmanNet)),
            MSE_train_KalmanNet,
            "-",
            color="b",
            markersize=10,
            label="MSE_train_KalmanNet",
        )
        plt.legend(prop={"family": "Times New Roman", "size": 24})  # SimSun
        plt.yticks(fontproperties="Times New Roman", size=24)
        plt.xticks(fontproperties="Times New Roman", size=24)
        plt.xlabel(
            "Training    Epoch", fontdict={"family": "Times New Roman", "size": 24}
        )
        plt.ylabel(
            "MSE    LOSS (dB)", fontdict={"family": "Times New Roman", "size": 24}
        )
        plt.savefig(fname="train_loss_compare.pdf", dpi=900, bbox_inches="tight")
        plt.show()

    def NNTrain(self, n_Examples, train_loader, valid_loader, n_CV):
        self.N_E = n_Examples
        self.N_CV = n_CV

        self.MSE_cv_linear_epoch = torch.empty(size=[self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])

        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        self.model.eval()

        for ti in tqdm(range(0, self.N_Epochs), desc="Training"):
            # 参数融合
            with torch.no_grad():
                epi = [10, 10, 10, 10]
                weights = []
                bias = []
                for sen in range(self.sensor_num):
                    weights.append(self.model.NN_list[sen].Info_l3_in.weight)
                    bias.append(self.model.NN_list[sen].Info_l3_in.bias)
                for i in range(self.sensor_num):
                    for j in range(self.sensor_num):
                        if i != j and self.model.Communication_m[i, j]:
                            self.model.NN_list[i].Info_l3_in.weight += (1 / epi[i]) * (
                                weights[j] - weights[i]
                            )
                            self.model.NN_list[i].Info_l3_in.bias += (1 / epi[i]) * (
                                bias[j] - bias[i]
                            )

            for train_index in range(len(self.model.NN_list)):

                # 待训练的网络计算梯度，不训练的网络不计算梯度
                for param in self.model.NN_list[train_index].parameters():
                    param.requires_grad = True
                for nTrain_index in range(len(self.model.NN_list)):
                    if nTrain_index != train_index:
                        for param in self.model.NN_list[nTrain_index].parameters():
                            param.requires_grad = False

                self.parameters = self.model.NN_list[train_index].parameters()
                self.optimizer = torch.optim.Adam(
                    self.parameters, lr=self.learningRate, weight_decay=self.weightDecay
                )
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=1e-3,
                    max_lr=1e-3,
                    step_size_up=20,
                    step_size_down=20,
                    cycle_momentum=False,
                )

                MSE_cv_linear_batch = []
                MSE_train_linear_batch = []

                self.model.NN_list[train_index].eval()
                for index, [cv_input, cv_target, cv_initial_state_all] in enumerate(
                    valid_loader
                ):
                    self.model.NN_list[train_index].init_hidden()
                    self.model.NN_list[train_index].InitSequence(
                        self.ssModel.m1x_0.repeat(self.N_B, 1, 1),
                        self.ssModel.m2x_0.repeat(self.N_B, 1, 1),
                        cv_initial_state_all,
                    )
                    [
                        self.Filters[i].InitSequence(
                            self.ssModel.Trans[i].repeat(self.N_B, 1, 1)
                            @ self.ssModel.m2x_0.repeat(self.N_B, 1, 1)
                            @ self.ssModel.Trans[i]
                            .repeat(self.N_B, 1, 1)
                            .transpose(1, 2),
                            self.ssModel.Trans[i].repeat(self.N_B, 1, 1)
                            @ self.ssModel.m1x_0.repeat(self.N_B, 1, 1),
                        )
                        for i in range(self.ssModel.sensor_num)
                    ]
                    x_out_cv = torch.zeros(self.N_B, self.ssModel.m, self.ssModel.T)
                    index_temp = 0
                    for t in range(0, self.ssModel.T):
                        x_prior = []
                        x_posterior = []
                        sigma_prior = []
                        sigma_posterior = []
                        index_temp = 0
                        for i in range(self.sensor_num):
                            yt = torch.unsqueeze(
                                cv_input[
                                    :, index_temp : index_temp + self.n_list[i], t
                                ],
                                -1,
                            )
                            index_temp += self.n_list[i]
                            (
                                x_prior_i,
                                x_posterior_i,
                                sigma_prior_i,
                                sigma_posterior_i,
                            ) = self.Filters[i].Update(yt)
                            x_prior.append(x_prior_i)
                            x_posterior.append(x_posterior_i)
                            sigma_prior.append(sigma_prior_i)
                            sigma_posterior.append(sigma_posterior_i)
                        xt, _, _, _ = self.model.NN_list[train_index](
                            x_prior,
                            x_posterior,
                            sigma_prior,
                            sigma_posterior,
                            self.model.Communication_m,
                        )
                        x_out_cv[:, :, t] = (
                            torch.pinverse(self.model.Trans[train_index])
                            @ xt.unsqueeze(-1)
                        ).squeeze(-1)

                    Loss = self.loss_fn(
                        self.ssModel.Trans[train_index].repeat(self.N_B, 1, 1)
                        @ x_out_cv,
                        self.ssModel.Trans[train_index].repeat(self.N_B, 1, 1)
                        @ cv_target,
                    )
                    MSE_cv_linear_batch.append(Loss.item())
                self.MSE_cv_linear_epoch[ti] = np.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(
                    self.MSE_cv_linear_epoch[ti]
                )

                if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, self.modelFileName)
                    filename = os.path.join(
                        "../Decentralized IFNet/model_state_dict/",
                        "checkpoint_epoch_" + str(ti) + ".pth.tar",
                    )
                    self.save_checkpoint(
                        {"epoch": ti, "backbone": self.model.state_dict()}, filename
                    )

                ###############################
                ### Training Sequence Batch ###
                ###############################
                self.model.train()
                self.model.NN_list[train_index].train()
                for index, [
                    train_input,
                    train_target,
                    train_initial_state_all,
                ] in enumerate(train_loader):
                    self.model.NN_list[train_index].init_hidden()
                    self.optimizer.zero_grad()
                    self.model.NN_list[train_index].InitSequence(
                        self.ssModel.m1x_0.repeat(self.N_B, 1, 1),
                        self.ssModel.m2x_0.repeat(self.N_B, 1, 1),
                        train_initial_state_all,
                    )
                    x_out_training = torch.zeros(
                        self.N_B, self.ssModel.m, self.ssModel.T
                    )
                    [
                        self.Filters[i].InitSequence(
                            self.ssModel.Trans[i].repeat(self.N_B, 1, 1)
                            @ self.ssModel.m2x_0.repeat(self.N_B, 1, 1)
                            @ self.ssModel.Trans[i]
                            .repeat(self.N_B, 1, 1)
                            .transpose(1, 2),
                            self.ssModel.Trans[i].repeat(self.N_B, 1, 1)
                            @ self.ssModel.m1x_0.repeat(self.N_B, 1, 1),
                        )
                        for i in range(self.ssModel.sensor_num)
                    ]
                    for t in range(0, self.ssModel.T):
                        x_prior = []
                        x_posterior = []
                        sigma_prior = []
                        sigma_posterior = []
                        index_temp = 0
                        for i in range(self.sensor_num):
                            yt = torch.unsqueeze(
                                train_input[
                                    :, index_temp : index_temp + self.n_list[i], t
                                ],
                                -1,
                            )
                            index_temp += self.n_list[i]
                            (
                                x_prior_i,
                                x_posterior_i,
                                sigma_prior_i,
                                sigma_posterior_i,
                            ) = self.Filters[i].Update(yt)
                            x_prior.append(x_prior_i)
                            x_posterior.append(x_posterior_i)
                            sigma_prior.append(sigma_prior_i)
                            sigma_posterior.append(sigma_posterior_i)
                        xt, _, _, _ = self.model.NN_list[train_index](
                            x_prior,
                            x_posterior,
                            sigma_prior,
                            sigma_posterior,
                            self.model.Communication_m,
                        )
                        x_out_training[:, :, t] = (
                            torch.pinverse(self.model.Trans[train_index])
                            @ xt.unsqueeze(-1)
                        ).squeeze(-1)
                    Loss = self.loss_fn(
                        self.ssModel.Trans[train_index].repeat(self.N_B, 1, 1)
                        @ x_out_training,
                        self.ssModel.Trans[train_index].repeat(self.N_B, 1, 1)
                        @ train_target,
                    )
                    MSE_train_linear_batch.append(Loss.item())
                    (Loss / self.N_B).backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.parameters, max_norm=100, norm_type=2
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                self.MSE_train_linear_epoch[ti] = np.mean(MSE_train_linear_batch)
                self.MSE_train_dB_epoch[ti] = 10 * torch.log10(
                    self.MSE_train_linear_epoch[ti]
                )
                tqdm.write(
                    str(ti)
                    + " "
                    + "MSE Training :"
                    + " "
                    + str(self.MSE_train_dB_epoch[ti])
                    + " "
                    + "[dB]"
                    + " "
                    + "MSE Validation :"
                    + " "
                    + str(self.MSE_cv_dB_epoch[ti])
                    + " "
                    + "[dB]"
                )
                if ti > 0:
                    d_train = (
                        self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                    )
                    d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                    tqdm.write(
                        "diff MSE Training :"
                        + " "
                        + str(d_train)
                        + "[dB]"
                        + " "
                        + "diff MSE Validation :"
                        + " "
                        + str(d_cv)
                        + " "
                        + "[dB]"
                    )

                    tqdm.write(
                        "Optimal idx :"
                        + " "
                        + str(self.MSE_cv_idx_opt)
                        + " "
                        + " Optimal :"
                        + " "
                        + str(self.MSE_cv_dB_opt)
                        + " "
                        + " [dB]"
                    )
                    torch.save(
                        [
                            self.MSE_train_linear_epoch,
                            self.MSE_train_dB_epoch,
                            self.MSE_cv_linear_epoch,
                            self.MSE_cv_dB_epoch,
                        ],
                        self.TrainLossFileName,
                    )
