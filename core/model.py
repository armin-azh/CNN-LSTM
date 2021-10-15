from torch import nn
from torch import Tensor
import torch

from settings import has_cuda

from torchsummary import summary


class CnnLSTM(nn.Module):
    def __init__(self, out_conv_filters: int, conv_kernel: int, conv_padding: str, pool_size: int, pool_padding: str,
                 lstm_hidden_unit: int, n_features: int, time_step: int):
        super(CnnLSTM, self).__init__()

        device = torch.device("cuda") if has_cuda else torch.device("cpu")

        self._conv = nn.Conv1d(in_channels=n_features, out_channels=out_conv_filters, kernel_size=conv_kernel,
                               padding=conv_padding, device=device)
        self._tanh = nn.Tanh()
        self._max_pool = nn.MaxPool1d(kernel_size=pool_size)
        self._relu = nn.ReLU(inplace=True)
        self._lstm = nn.LSTM(batch_first=True, hidden_size=lstm_hidden_unit, input_size=time_step, num_layers=1,
                             bidirectional=False,
                             device=device)
        self._lstm2 = nn.LSTM(batch_first=True, hidden_size=lstm_hidden_unit, input_size=lstm_hidden_unit, num_layers=1,
                              bidirectional=False,
                              device=device)
        self._linear = nn.Linear(in_features=lstm_hidden_unit * out_conv_filters, out_features=1, device=device)
        self._flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        x = self._conv(x)
        x = self._tanh(x)
        x = self._max_pool(x)
        x = self._relu(x)

        x, (hn, cn) = self._lstm(x)
        # x, (_, _) = self._lstm2(x)
        x = x.reshape((len(x), -1))
        x = self._linear(x)
        return x

# if __name__ == '__main__':
#     model = CnnLSTM(out_conv_filters=32, conv_kernel=1, conv_padding="same", pool_padding="same", pool_size=1,
#                     lstm_hidden_unit=64, n_features=7)
#
#     summary(model=model, input_data=(7, 10))
