import warnings

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from settings import label_column_name


class StockPriceDataset(Dataset):
    def __init__(self, filepath: str, time_step, test: bool = False, train_size: float = 0.7, test_size: float = 0.3,
                 phase: str = "train"):
        data_df = pd.read_csv(filepath)
        data_df = data_df.drop(labels=np.where(data_df.isnull().any(axis=1) == True)[0], axis=0)
        data_df = data_df.drop(labels="Adj Close", axis=1)
        data_df = data_df.loc[(data_df["Volume"] != "0") & (data_df["Volume"] != 0)]

        if phase == "train":
            train_df, valid_df = train_test_split(data_df, test_size=test_size, train_size=train_size, shuffle=False)
            if not test:
                std = train_df.std()
                mean = train_df.mean()

                std_open = std.Open
                std_high = std.High
                std_low = std.Low
                std_close = std.Close
                std_volume = std.Volume

                mean_open = mean.Open
                mean_high = mean.High
                mean_low = mean.Low
                mean_close = mean.Close
                mean_volume = mean.Volume


            else:
                pass
        elif phase == "test":
            pass
        else:
            print("[Failed] You had selected wrong phase")


if __name__ == '__main__':
    filename = "/home/lezarus/Documents/Project/cnn_lstm/data/dataset/000001.SS.csv"
    dataset = StockPriceDataset(filepath=filename, time_step=10)
