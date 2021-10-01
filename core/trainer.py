from typing import Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

from core.model import CnnLSTM

from settings import has_cuda

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


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
        self._criterion = MSELoss()

    def train(self, train_loader: DataLoader, epochs: int, test_loader: Union[DataLoader, None], save_path: Path):
        print(f'[Train] Start to train')

        total_loss = []
        # total_acc = []

        # start update
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

        # end update

        total_loss = np.array(total_loss)
        np.save(file=str(save_path.joinpath("loss.npy")), arr=total_loss)
        torch.save(self._model.state_dict(), str(save_path.joinpath("model.pth")))

        # start save loss plot
        plt.figure(figsize=(15, 9))
        plt.plot(np.arange(epochs) + 1, total_loss)
        plt.title("CNN-LSTM", fontsize=18, fontweight='bold')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.savefig(str(save_path.joinpath("plot").joinpath("loss.png")))
        # end save loss plot

        # start test
        # end test
