from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss
from torch.optim.lr_scheduler import ExponentialLR

from core.model import CnnLSTM

from settings import has_cuda
from core.loss import *

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class CnnLstmTrainer:
    def __init__(self,time_step:int, out_conv_filters: int, conv_kernel: int, conv_padding: str, pool_size: int, pool_padding: str,
                 lstm_hidden_unit: int, n_features: int, lr: float, loss: Union[L1Loss, RMSELoss, RLoss]):

        print(f"[Model] Loading model")
        self._model = CnnLSTM(out_conv_filters, conv_kernel, conv_padding, pool_size, pool_padding, lstm_hidden_unit,
                              n_features,time_step=time_step)
        print(f"[Model] model had been loaded")

        if has_cuda:
            self._model.cuda()

        self._opt = Adam(self._model.parameters(), lr=lr)
        self._criterion = loss()
        self._criterion_name = self._criterion.__class__.__name__
        self._scheduler = ExponentialLR(self._opt, gamma=0.9)

    def train(self, train_loader: DataLoader, epochs: int, validation_loader: DataLoader,
              test_loader: Union[DataLoader, None], save_path: Path,
              scale: dict):
        print(f'[Train] Start to train')

        std = scale["std"]
        mean = scale["mean"]

        total_loss = []
        total_val_loss = []
        # total_acc = []

        # start update
        step = 0
        for epoch in range(epochs):

            tm_loss = []
            tm_val_loss = []
            # tm_acc = []

            # training proccess
            for idx, (x, y) in enumerate(train_loader):

                x = torch.transpose(x, dim0=1, dim1=2)

                # x_mean = torch.mean(x)
                # y_mean = torch.mean(y)
                #
                # x_std = torch.std(x)
                # y_std = torch.std(y)
                #
                # x = (x-x_mean)/x_std
                # y = (y-y_mean)/y_std

                if has_cuda:
                    x = x.float().cuda()
                    y = y.float().cuda()

                # # update
                self._opt.zero_grad()
                pred = self._model(x)
                # pred = pred*std+mean
                # y = y * std + mean
                loss = self._criterion(pred, y)
                loss.backward()
                self._opt.step()
                tm_loss.append(loss.item())
                if step % 150 == 0:
                    print(f"[{epoch + 1}/{epochs}] Epoch | Loss:{tm_loss[-1]}")
                step += 1

            # self._scheduler.step()
            total_loss.append(np.mean(tm_loss))

            # start validation process
            for idx, (x, y) in enumerate(validation_loader):
                x = torch.transpose(x, dim0=1, dim1=2)

                # x_mean = torch.mean(x)
                # y_mean = torch.mean(y)
                #
                # x_std = torch.std(x)
                # y_std = torch.std(y)
                #
                # x = (x-x_mean)/x_std
                # y = (y-y_mean)/y_std

                if has_cuda:
                    x = x.float().cuda()
                    y = y.float().cuda()

                with torch.no_grad():
                    pred = self._model(x)
                    # pred = pred * std + mean
                    # y = y * std + mean
                    loss = self._criterion(pred, y)
                    tm_val_loss.append(loss.item())
            total_val_loss.append(np.mean(tm_val_loss))
            print(f"[{epoch + 1}/{epochs}] Epoch | Validation Loss:{np.array(total_val_loss).mean()}")
            # end validation process

        # end update

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # start saving weights
        total_loss = np.array(total_loss)
        total_val_loss = np.array(total_val_loss)
        total_loss = 1 - total_loss if self._criterion_name == "RLoss" else total_loss
        total_val_loss = 1 - total_val_loss if self._criterion_name == "RLoss" else total_val_loss

        np.save(file=str(save_path.joinpath("train_loss.npy")), arr=total_loss)
        np.save(file=str(save_path.joinpath("validation_loss.npy")), arr=total_val_loss)
        torch.save(self._model.state_dict(), str(save_path.joinpath("model.pth")))
        # end saving weights

        # start save loss plot
        plt.figure(figsize=(15, 9))
        plt.plot(np.arange(epochs) + 1, total_loss, color=colors[0], label="Train loss")
        plt.plot(np.arange(epochs) + 1, total_val_loss, color=colors[1], label="Validation loss")
        plt.title(f"CNN-LSTM ({self._criterion_name})", fontsize=18, fontweight='bold')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.legend()
        plt.savefig(str(save_path.joinpath("plot").joinpath("training_loss.png")))
        # end save loss plot

        # start test

        test_loss = []
        val_pred = []
        val_true = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = torch.transpose(x, dim0=1, dim1=2)

                # x_mean = torch.mean(x)
                # y_mean = torch.mean(y)
                #
                # x_std = torch.std(x)
                # y_std = torch.std(y)
                #
                # x = (x-x_mean)/x_std
                # y = (y-y_mean)/y_std

                if has_cuda:
                    x = x.float().cuda()
                    y = y.float().cuda()

                # # update
                pred = self._model(x)
                pred = pred * std + mean
                y = y * std + mean
                loss = self._criterion(pred, y)

                val_pred.append(pred.cpu().numpy())
                val_true.append(y.cpu().numpy())
                test_loss.append(loss.item())

        val_pred = np.concatenate(val_pred, axis=0)
        val_true = np.concatenate(val_true, axis=0)

        val_pred = np.squeeze(val_pred, axis=-1)
        val_true = np.squeeze(val_true, axis=-1)

        # val_pred = np.squeeze(val_pred , axis=-1)
        # val_true = np.squeeze(val_true, axis=-1)

        test_loss_mean = np.array(test_loss).mean()
        test_loss_mean = 1 - test_loss_mean if self._criterion_name == "RLoss" else test_loss_mean

        s_val_pred = np.array(val_pred)
        s_val_true = np.array(val_true)

        np.save(str(save_path.joinpath("train_pred.npy")), s_val_pred)
        np.save(str(save_path.joinpath("train_true.npy")), s_val_true)

        predict = pd.DataFrame(val_pred)
        original = pd.DataFrame(val_true)

        # start save prediction
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(x=original.index, y=original[0], label="Data", color='royalblue')
        ax = sns.lineplot(x=predict.index, y=predict[0], label=f"Prediction", color='tomato')
        ax.set_title(f'Test Stock price (Test loss: {test_loss_mean})', size=14, fontweight='bold')
        ax.set_xlabel("Days", size=14)
        ax.set_ylabel("Cost (USD)", size=14)
        ax.set_xticklabels('', size=10)

        plt.savefig(str(save_path.joinpath("plot").joinpath("prediction.png")))
        # end save loss plot

        # end test
        print(f"[Test] Final Test Loss: {test_loss_mean}")
