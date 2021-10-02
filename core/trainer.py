from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss

from core.model import CnnLSTM

from settings import has_cuda
from core.loss import *

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class CnnLstmTrainer:
    def __init__(self, out_conv_filters: int, conv_kernel: int, conv_padding: str, pool_size: int, pool_padding: str,
                 lstm_hidden_unit: int, n_features: int, lr: float, loss: Union[L1Loss, RMSELoss, RLoss]):

        print(f"[Model] Loading model")
        self._model = CnnLSTM(out_conv_filters, conv_kernel, conv_padding, pool_size, pool_padding, lstm_hidden_unit,
                              n_features)
        print(f"[Model] model had been loaded")

        if has_cuda:
            self._model.cuda()

        self._opt = Adam(self._model.parameters(), lr=lr)
        self._criterion = loss()
        self._criterion_name = self._criterion.__class__.__name__

    def train(self, train_loader: DataLoader, epochs: int, test_loader: Union[DataLoader, None], save_path: Path,
              scale: dict):
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
        plt.title(f"CNN-LSTM ({self._criterion_name})", fontsize=18, fontweight='bold')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.savefig(str(save_path.joinpath("plot").joinpath("loss.png")))
        # end save loss plot

        # start test

        test_loss = []
        val_pred = []
        val_true = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = torch.transpose(x, dim0=1, dim1=2)

                if has_cuda:
                    x = x.float().cuda()
                    y = y.float().cuda()

                # # update
                pred = self._model(x)

                loss = self._criterion(pred, y)

                val_pred.append(pred.cpu().numpy())
                val_true.append(y.cpu().numpy())
                test_loss.append(loss.item())

        # start save loss plot
        test_loss = np.array(test_loss)
        plt.figure(figsize=(15, 9))
        plt.plot(np.arange(len(test_loss)) + 1, test_loss)
        plt.title(f"Test CNN-LSTM Loss ({self._criterion_name})", fontsize=18, fontweight='bold')
        plt.xlabel('# Batches', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.savefig(str(save_path.joinpath("plot").joinpath("test_loss.png")))
        # end save loss plot

        val_pred = np.concatenate(val_pred, axis=0)
        val_true = np.concatenate(val_true, axis=0)

        std = scale["std"]
        mean = scale["mean"]

        val_pred = np.squeeze(std * val_pred + mean, axis=-1)
        val_true = np.squeeze(std * val_true + mean, axis=-1)

        predict = pd.DataFrame(val_pred)
        original = pd.DataFrame(val_true)

        # start save prediction
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(x=original.index, y=original[0], label="Data", color='royalblue')
        ax = sns.lineplot(x=predict.index, y=predict[0], label="Training Prediction (CNN-LSTM)", color='tomato')
        ax.set_title('Test Stock price', size=14, fontweight='bold')
        ax.set_xlabel("Days", size=14)
        ax.set_ylabel("Cost (USD)", size=14)
        ax.set_xticklabels('', size=10)

        plt.savefig(str(save_path.joinpath("plot").joinpath("prediction.png")))
        # end save loss plot

        # end test
        print(f"[Test] Final Test Loss: {test_loss[-1]}")
