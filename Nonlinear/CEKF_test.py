import torch
import torch.nn as nn
import time
from CEKF import CEKF
from Utils.Extended_data import N_T


def CEKFTest(
    SysModel, test_input, test_target, sensor_num, Trans, position_idx, velocity_idx
):
    # LOSS
    loss_fn = nn.MSELoss(reduction="mean")

    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(N_T)

    EKF = CEKF(SysModel)
    EKF_out = torch.zeros_like(test_target)
    time_cost = time.perf_counter()
    for i in range(0, N_T):
        EKF.InitSequence(SysModel.m2x_0, SysModel.initial_state_all["test"][i, :, :])
        EKF.GenerateSequence(test_input[i, :, :], EKF.T_test)
        EKF_out[i, :, :] = EKF.x
        MSE_KF_linear_arr[i] = loss_fn(EKF.x, test_target[i, :, :]).item()
    time_cost = time.perf_counter() - time_cost

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    EKF_out = EKF_out.permute(1, 0, 2)
    test_target = test_target.permute(1, 0, 2)
    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    MSE_seq = [torch.zeros((test_target.size()[2], 1)) for _ in range(sensor_num)]
    MSE_seq_pos = [torch.zeros((test_target.size()[2], 1)) for _ in range(sensor_num)]
    MSE_seq_vel = [torch.zeros((test_target.size()[2], 1)) for _ in range(sensor_num)]
    for j in range(sensor_num):
        for i in range(test_target.size()[2]):
            MSE_seq[j][i] = torch.sqrt(
                torch.mean(
                    (Trans[j] @ EKF_out[:, :, i] - Trans[j] @ test_target[:, :, i]) ** 2
                )
            )
            MSE_seq_pos[j][i] = torch.sqrt(
                torch.mean(
                    (
                        (Trans[j] @ EKF_out[:, :, i])[position_idx[j], :]
                        - (Trans[j] @ test_target[:, :, i])[position_idx[j], :]
                    )
                    ** 2
                )
            )
            MSE_seq_vel[j][i] = torch.sqrt(
                torch.mean(
                    (
                        (Trans[j] @ EKF_out[:, :, i])[velocity_idx[j], :]
                        - (Trans[j] @ test_target[:, :, i])[velocity_idx[j], :]
                    )
                    ** 2
                )
            )
        # MSE_seq[i]=torch.sqrt(torch.mean((KF_out[:,state_idx[0],i]-test_target[:,state_idx[0],i])**2+(KF_out[:,state_idx[1],i]-test_target[:,state_idx[1],i])**2))
    return [
        MSE_KF_linear_arr,
        MSE_KF_linear_avg,
        MSE_KF_dB_avg,
        EKF_out,
        MSE_seq,
        MSE_seq_pos,
        MSE_seq_vel,
        time_cost,
    ]
