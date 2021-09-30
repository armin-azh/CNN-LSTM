import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from settings import label_column_name


class StockPriceDataset(Dataset):
    def __init__(self, filepath: str, time_step, test: bool = False, train_size: float = 0.7, test_size: float = 0.3,
                 phase: str = "train"):

        self._data = None
        self._data_df = None
        data_df = pd.read_csv(filepath)
        data_df = data_df.drop(labels=np.where(data_df.isnull().any(axis=1) == True)[0], axis=0)
        data_df = data_df.drop(labels=["Date", "Adj Close"], axis=1)
        data_df = data_df.loc[(data_df["Volume"] != "0") & (data_df["Volume"] != 0)]

        if phase == "train":
            train_df, valid_df = train_test_split(data_df, test_size=test_size, train_size=train_size, shuffle=False)
            if not test:
                self._data_df = train_df
            else:
                self._data_df = valid_df
        elif phase == "test":
            self._data_df = data_df
        else:
            print("[Failed] You had selected wrong phase")

        # normalize and convert to numpy
        (self._std_open, self._std_high, self._std_low, self._std_close, self._std_volume), (
            self._mean_open, self._mean_high, self._mean_low, self._mean_close, self._mean_volume) = self.ds_state(
            self._data_df)

        self._data_df.loc[:, "Open"] = self.z_score(self._data_df.loc[:, "Open"], self._std_open, self._mean_open)
        self._data_df.loc[:, "High"] = self.z_score(self._data_df.loc[:, "High"], self._std_high, self._mean_high)
        self._data_df.loc[:, "Low"] = self.z_score(self._data_df.loc[:, "Low"], self._std_low, self._mean_low)
        self._data_df.loc[:, "Close"] = self.z_score(self._data_df.loc[:, "Close"], self._std_close, self._mean_close)
        self._data_df.loc[:, "Volume"] = self.z_score(self._data_df.loc[:, "Volume"], self._std_volume,
                                                      self._mean_volume)

        self._data = self._data_df.copy().to_numpy()

        # fix the tensor
        n_sample = self._data.shape[0]
        c_sample = n_sample // time_step
        r_sample = n_sample % time_step

        c_temp = self._data[:c_sample * time_step, ...].reshape((-1, time_step, 5))
        if r_sample > 0:
            r_temp = self._data[n_sample - time_step:, ...].reshape(-1, time_step, 5)
            c_temp = np.concatenate([c_temp, r_temp], axis=0)

        self._data = c_temp

    @staticmethod
    def z_score(col_val, std_val, mean_val):
        return (col_val - mean_val) / std_val

    @staticmethod
    def ds_state(ds_df):
        std = ds_df.std()
        mean = ds_df.mean()

        return (std.Open, std.High, std.Low, std.Close, std.Volume), (
            mean.Open, mean.High, mean.Low, mean.Close, mean.Volume)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._data[idx, ...]


if __name__ == '__main__':
    filename = "/home/lezarus/Documents/Project/cnn_lstm/data/dataset/000001.SS.csv"
    dataset = StockPriceDataset(filepath=filename, time_step=10)

