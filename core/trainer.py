from typing import Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss

from core.model import CnnLSTM

from settings import has_cuda


class CnnLstmTrainer:
    def __init__(self, out_conv_filters: int, conv_kernel: int, conv_padding: str, pool_size: int, pool_padding: str,
                 lstm_hidden_unit: int, n_features: int, lr: float):

        print(f"[Model] Loading model")
        self._model = CnnLSTM(out_conv_filters, conv_kernel, conv_padding, pool_size, pool_padding, lstm_hidden_unit,
                              n_features)
        print(f"[Model] model had been loaded")

        if has_cuda:
            self._model.cuda()

        self._opt = Adam(self._model.parameters(), lr=lr)
        self._criterion = L1Loss()

    def train(self, train_loader: DataLoader, epochs: int, test_loader: Union[DataLoader, None]):
        print(f'[Train] Start to train')

        total_loss = []
        # total_acc = []

        for epoch in range(epochs):

            tm_loss = []
            # tm_acc = []

            for idx, (x, y) in enumerate(train_loader):
                x = torch.transpose(x, dim0=1, dim1=2)

                if has_cuda:

                    x = x.float().cuda()
                    y = y.float().cuda()

                # # update
                self._opt.zero_grad()
                pred = self._model(x)

                loss = self._criterion(pred, y)
                loss.backward()
                self._opt.step()
                tm_loss.append(loss.item())
            total_loss.append(np.mean(tm_loss))
            if epoch % 10 == 0 and epoch > 0:
                print(f"[{epoch + 1}/{epochs}] Epoch | Loss:{total_loss[-1]}")
