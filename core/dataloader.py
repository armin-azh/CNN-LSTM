import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from settings import label_column_name


class StockPriceDataset(Dataset):
    def __init__(self, filepath: str, time_step, test: bool = False, train_size: float = 0.7, test_size: float = 0.3,
                 phase: str = "train"):

        self._data = None
        data_df = pd.read_csv(filepath)
        data_df = data_df.drop(labels=np.where(data_df.isnull().any(axis=1) == True)[0], axis=0)
        data_df = data_df.drop(labels=["Date", "Adj Close"], axis=1)
        data_df = data_df.loc[(data_df["Volume"] != "0") & (data_df["Volume"] != 0)]

        if phase == "train":
            train_df, valid_df = train_test_split(data_df, test_size=test_size, train_size=train_size, shuffle=False)
            if not test:
                (std_open, std_high, std_low, std_close, std_volume), (
                    mean_open, mean_high, mean_low, mean_close, mean_volume) = self.ds_state(train_df)

                train_df.loc[:, "Open"] = self.z_score(train_df.loc[:, "Open"], std_open, mean_open)
                train_df.loc[:, "High"] = self.z_score(train_df.loc[:, "High"], std_high, mean_high)
                train_df.loc[:, "Low"] = self.z_score(train_df.loc[:, "Low"], std_low, mean_low)
                train_df.loc[:, "Close"] = self.z_score(train_df.loc[:, "Close"], std_close, mean_close)
                train_df.loc[:, "Volume"] = self.z_score(train_df.loc[:, "Volume"], std_volume, mean_volume)

                self._data = train_df.copy()

                print(self._data.to_numpy().shape)

            else:
                pass
        elif phase == "test":
            pass
        else:
            print("[Failed] You had selected wrong phase")

    @staticmethod
    def z_score(col_val, std_val, mean_val):
        return (col_val - mean_val) / std_val

    @staticmethod
    def ds_state(ds_df):
        std = ds_df.std()
        mean = ds_df.mean()

        return (std.Open, std.High, std.Low, std.Close, std.Volume), (
            mean.Open, mean.High, mean.Low, mean.Close, mean.Volume)

    @staticmethod
    def to_numpy(ds_df):
        pass


if __name__ == '__main__':
    filename = "/home/lezarus/Documents/Project/cnn_lstm/data/dataset/000001.SS.csv"
    dataset = StockPriceDataset(filepath=filename, time_step=10)
