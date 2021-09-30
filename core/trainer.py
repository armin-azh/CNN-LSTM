from typing import Union
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss

from core.model import CnnLSTM


class CnnLstmTrainer:
    def __init__(self, out_conv_filters: int, conv_kernel: int, conv_padding: str, pool_size: int, pool_padding: str,
                 lstm_hidden_unit: int, n_features: int, lr: float):
        self._model = CnnLSTM(out_conv_filters, conv_kernel, conv_padding, pool_size, pool_padding, lstm_hidden_unit,
                              n_features)

        self._opt = Adam(self._model.parameters(), lr=lr)
        self.criterion = L1Loss()

    def train(self, train_loader: DataLoader, epochs: int, test_loader: Union[DataLoader, None]):
        for idx, (x, y) in enumerate(train_loader):
            pass
