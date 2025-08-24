import torch
import torch.nn as nn
import time

from Extended_DIF import Extended_DIF
from Utils.Extended_data import N_T


def Extended_DIFTest(
    SysModel,
    SysModel_EKF,
    test_input,
    test_target,
    position_idx,
    velocity_idx,
    sensor_num,
):

    # LOSS
    loss_fn = nn.MSELoss(reduction="mean")

    # MSE [Linear]
    MSE_DIF_linear_arr = [torch.zeros(N_T) for _ in range(sensor_num)]

    EDIF = Extended_DIF(SysModel, SysModel_EKF)

    EDIF_out = [torch.zeros_like(test_target) for _ in range(sensor_num)]
    time_cost_total = 0
    for j in range(0, N_T):
        EDIF.InitSequence(
            SysModel.m1x_0, SysModel.m2x_0, SysModel.initial_state_all["test"][j, :, :]
        )

        EDIF.GenerateSequence(test_input[j, :, :], EDIF.T_test)
        time_cost_total += EDIF.time_cost_total
        for i in range(sensor_num):
            EDIF_out[i][j, :, :] = EDIF.x[i]
            MSE_DIF_linear_arr[i][j] = loss_fn(
                SysModel.Trans[i] @ EDIF.x[i], SysModel.Trans[i] @ test_target[j, :, :]
            ).item()
    MSE_DIF_linear_avg = []
    MSE_DIF_dB_avg = []
    MSE_seq = [torch.zeros((test_target.size()[2], 1)) for _ in range(sensor_num)]
    MSE_seq_pos = [torch.zeros((test_target.size()[2], 1)) for _ in range(sensor_num)]
    MSE_seq_vel = [torch.zeros((test_target.size()[2], 1)) for _ in range(sensor_num)]
    print(end="\n")
    print("Decentralized Information Filter")
    for i in range(sensor_num):
        MSE_DIF_linear_avg.append(torch.mean(MSE_DIF_linear_arr[i]))
        MSE_DIF_dB_avg.append(10 * torch.log10(MSE_DIF_linear_avg[i]))
        # DIF_out[i][:, :, 0] = torch.squeeze(SysModel.initial_state_all["test"][:, :, :])
        print("Sensor" + str(i + 1) + "'s MSE", MSE_DIF_dB_avg[i], "[dB]", end="\t")
        adjusted_DIF = EDIF_out[i].permute(1, 0, 2)
        adjusted_target = test_target.permute(1, 0, 2)
        for j in range(test_target.size()[2]):
            MSE_seq[i][j] = torch.sqrt(
                torch.mean(
                    (
                        SysModel.Trans[i] @ adjusted_DIF[:, :, j]
                        - SysModel.Trans[i] @ adjusted_target[:, :, j]
                    )
                    ** 2
                )
            )
            MSE_seq_pos[i][j] = torch.sqrt(
                torch.mean(
                    (
                        (SysModel.Trans[i] @ adjusted_DIF[:, :, j])[position_idx[i], :]
                        - (SysModel.Trans[i] @ adjusted_target[:, :, j])[
                            position_idx[i], :
                        ]
                    )
                    ** 2
                )
            )
            MSE_seq_vel[i][j] = torch.sqrt(
                torch.mean(
                    (
                        (SysModel.Trans[i] @ adjusted_DIF[:, :, j])[velocity_idx[i], :]
                        - (SysModel.Trans[i] @ adjusted_target[:, :, j])[
                            velocity_idx[i], :
                        ]
                    )
                    ** 2
                )
            )
    return [
        MSE_DIF_linear_arr,
        MSE_DIF_linear_avg,
        MSE_DIF_dB_avg,
        EDIF_out,
        MSE_seq,
        MSE_seq_pos,
        MSE_seq_vel,
        time_cost_total,
    ]
