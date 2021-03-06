import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter("ignore", UserWarning)

from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class StockPriceDataset(Dataset):
    def __init__(self, filepath: str, time_step, save_plot: Union[Path, None],
                 train: bool,
                 validation: bool,
                 col_name: str,
                 train_size: float = 0.7,
                 test_size: float = 0.3,
                 phase: str = "train"):

        self._data = None
        self._true = None
        self._data_df = None
        self._scale = MinMaxScaler(feature_range=(-1, 1))
        data_df = pd.read_csv(filepath)
        data_df = data_df.drop(labels=np.where(data_df.isnull().any(axis=1) == True)[0], axis=0)

        if train:
            plt.figure(figsize=(15, 9))
            plt.plot(data_df[[col_name]])
            plt.xticks(range(0, data_df.shape[0], 500), data_df['Date'].loc[::500], rotation=45)
            plt.title("Stock Price", fontsize=18, fontweight='bold')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close Price (USD)', fontsize=18)
            plt.savefig(str(save_plot.joinpath("whole_data.png")))

        data_df = data_df.loc[(data_df["Volume"] != "0") & (data_df["Volume"] != 0)]

        # normalize and convert to numpy
        (self._std_open, self._std_high, self._std_low, self._std_close, self._std_volume, self._std_up,
         self._std_change), (
            self._mean_open, self._mean_high, self._mean_low, self._mean_close, self._mean_volume, self._mean_up,
            self._mean_change) = self.ds_state(data_df)

        if phase == "train":
            train_df = data_df.iloc[:-210]
            test_df = data_df.iloc[-210:]
            train_df, valid_df = train_test_split(train_df, test_size=test_size, train_size=train_size, shuffle=False)

            if train:

                plt.figure(figsize=(15, 9))
                plt.plot(train_df[[col_name]])
                plt.xticks(range(0, train_df.shape[0], 500), train_df['Date'].loc[::500], rotation=45)
                plt.title("Train Stock Price", fontsize=18, fontweight='bold')
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price (USD)', fontsize=18)
                plt.savefig(str(save_plot.joinpath("train_data.png")))

                self._data_df = train_df

            elif validation:
                plt.figure(figsize=(15, 9))
                plt.plot(valid_df[[col_name]])
                plt.xticks(range(0, valid_df.shape[0], 500), valid_df['Date'].loc[::500], rotation=45)
                plt.title("Validation Stock Price", fontsize=18, fontweight='bold')
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price (USD)', fontsize=18)
                plt.savefig(str(save_plot.joinpath("validation_data.png")))

                self._data_df = valid_df

            else:

                plt.figure(figsize=(15, 9))
                plt.plot(test_df[[col_name]])
                plt.xticks(range(0, test_df.shape[0], 500), test_df['Date'].loc[::500], rotation=45)
                plt.title("Test Stock Price", fontsize=18, fontweight='bold')
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price (USD)', fontsize=18)
                plt.savefig(str(save_plot.joinpath("test_data.png")))
                self._data_df = test_df

        elif phase == "test":
            self._data_df = data_df
        else:
            print("[Failed] You had selected wrong phase")

        self._data_df.loc[:, "Open"] = self.z_score(self._data_df.loc[:, "Open"], self._std_open, self._mean_open)
        self._data_df.loc[:, "High"] = self.z_score(self._data_df.loc[:, "High"], self._std_high, self._mean_high)
        self._data_df.loc[:, "Low"] = self.z_score(self._data_df.loc[:, "Low"], self._std_low, self._mean_low)
        self._data_df.loc[:, "Close"] = self.z_score(self._data_df.loc[:, "Close"], self._std_close, self._mean_close)
        self._data_df.loc[:, "Volume"] = self.z_score(self._data_df.loc[:, "Volume"], self._std_volume,
                                                      self._mean_volume)
        self._data_df.loc[:, "Ups and Downs"] = self.z_score(self._data_df.loc[:, "Ups and Downs"], self._std_up,
                                                             self._mean_up)
        self._data_df.loc[:, "Changes"] = self.z_score(self._data_df.loc[:, "Changes"], self._std_change,
                                                       self._mean_change)

        self._data_df = self._data_df.iloc[:, 1:]
        self._true = self._data_df.loc[:, col_name].to_numpy()
        self._data = self._data_df.copy().to_numpy()

        # fix the tensor

        tm_data = []
        tm_label = []

        for idx in range(len(self._data) - time_step):
            tm_data.append(self._data[idx:idx + time_step, ...])
            tm_label.append(self._true[idx + time_step])

        self._data = np.array(tm_data)
        self._true = np.array(tm_label)

    @staticmethod
    def z_score(col_val, std_val, mean_val):
        return (col_val - mean_val) / std_val

    @staticmethod
    def ds_state(ds_df):
        std = ds_df.std()
        mean = ds_df.mean()

        return (std.Open, std.High, std.Low, std.Close, std.Volume, std["Ups and Downs"], std.Changes), (
            mean.Open, mean.High, mean.Low, mean.Close, mean.Volume, mean["Ups and Downs"], mean.Changes)

    @property
    def std_scale(self):
        return self._std_close

    @property
    def mean_scale(self):
        return self._mean_close

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.from_numpy(self._data[idx, ...]), torch.from_numpy(np.expand_dims(self._true[idx, ...], axis=-1))

# if __name__ == '__main__':
#     save_path = Path("/home/lezarus/Documents/Project/cnn_lstm/result")
#     filename = "/home/lezarus/Documents/Project/cnn_lstm/data/dataset/aggregated.csv"
#     dataset = StockPriceDataset(filepath=filename, time_step=10, save_plot=save_path, train=True, validation=False)
#
#     print(dataset[0][0])
#     x = torch.transpose(dataset[0][0], dim0=0, dim1=1)
#     print(x)
#
#     print(dataset[0][0].shape)
#     print(x.shape)
