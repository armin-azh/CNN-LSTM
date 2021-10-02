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
        n = target.shape[0]
        sum_reg = torch.sum(torch.pow(torch.sub(target, y_hat), 2))
        sum_tot = torch.sum(torch.pow(torch.sub(target, torch.mean(target)), 2))
        return torch.div(sum_reg, sum_tot)


LOSS_FACTORY = {
    "mae": nn.L1Loss,
    "rmse": RMSELoss,
    "r": RLoss
}
