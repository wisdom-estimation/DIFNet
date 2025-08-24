import torch
import os
from torch.utils.data import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


N_E = 100
N_CV = 20
N_T = 40

T = 100
T_test = 100


def DataGen(SysModel_data, fileName, T, T_test, randomInit=False):
    print("Generate Training Sequence")
    SysModel_data.GenerateBatch(N_E, T, randomInit=randomInit)
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target

    print("Generate Validation Sequence")
    SysModel_data.GenerateBatch(N_CV, T, randomInit=randomInit)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target

    print("Generate Test Sequence")
    SysModel_data.GenerateBatch(N_T, T_test, randomInit=randomInit)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    torch.save(
        [training_input, training_target, cv_input, cv_target, test_input, test_target],
        fileName,
    )


def DataLoader_CPU(fileName):

    [training_input, training_target, cv_input, cv_target, test_input, test_target] = (
        torch.load(fileName, map_location=torch.device("cpu"))
    )
    return [
        training_input,
        training_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ]


def DataLoader_GPU(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = (
        torch.utils.data.DataLoader(torch.load(fileName), pin_memory=False)
    )
    training_input = training_input.squeeze().to(torch.device("cuda:0"))
    training_target = training_target.squeeze().to(torch.device("cuda:0"))
    cv_input = cv_input.squeeze().to(torch.device("cuda:0"))
    cv_target = cv_target.squeeze().to(torch.device("cuda:0"))
    test_input = test_input.squeeze().to(torch.device("cuda:0"))
    test_target = test_target.squeeze().to(torch.device("cuda:0"))
    return [
        training_input,
        training_target,
        cv_input,
        cv_target,
        test_input,
        test_target,
    ]


def log_record(logger, data, explain=None):
    if explain == None:
        logger.info(data)
        logger.info("\n")
    else:
        logger.info(f"{explain}:")
        logger.info(data)
        logger.info("\n")

    for handler in logger.handlers:
        handler.flush()


class My_Dataset(Dataset):
    def __init__(self, data, targets, initial_state_all):
        self.sensor_data = data
        self.sensor_targets = targets
        self.sensor_initial_state_all = initial_state_all

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, idx):
        x = self.sensor_data[idx]
        y = self.sensor_targets[idx]
        s = self.sensor_initial_state_all[idx]
        return x, y, s
