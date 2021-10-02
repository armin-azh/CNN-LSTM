from torch import nn
import torch


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self._mse = nn.MSELoss()

    def forward(self, y_hat, target):
        return torch.sqrt(self._mse(y_hat, target))


class RLoss(nn.Module):
    def __init__(self):
        super(RLoss, self).__init__()

    def forward(self, y_hat, target):
        target_mean = torch.mean(target)
        sum_tot = torch.sum((target - target_mean) ** 2)
        sum_res = torch.sum((target - y_hat) ** 2)
        return 1 - (sum_res / sum_tot)


LOSS_FACTORY = {
    "mae": nn.L1Loss,
    "rmse": RMSELoss,
    "r": RLoss
}
