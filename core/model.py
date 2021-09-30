from torch import nn
from torch import Tensor
import torch

from settings import has_cuda

from torchsummary import summary


class CnnLSTM(nn.Module):
    def __init__(self, out_conv_filters: int, conv_kernel: int, conv_padding: str, pool_size: int, pool_padding: str,
                 lstm_hidden_unit: int, n_features: int):
        super(CnnLSTM, self).__init__()

        device = torch.device("cuda") if has_cuda else torch.device("cpu")

        self._conv = nn.Conv1d(in_channels=n_features, out_channels=out_conv_filters, kernel_size=conv_kernel,
                               padding=conv_padding, device=device)
        self._tanh = nn.Tanh()
        self._max_pool = nn.MaxPool1d(kernel_size=pool_size)
        self._relu = nn.ReLU(inplace=True)
        self._lstm = nn.LSTM(batch_first=True, hidden_size=lstm_hidden_unit, input_size=10, num_layers=1,
                             bidirectional=False,
                             device=device)
        self._linear = nn.Linear(in_features=32 * 64, out_features=1, device=device)
        self._flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        x = self._conv(x)
        x = self._tanh(x)
        x = self._max_pool(x)
        x = self._relu(x)
        x = self._lstm(x)
        x = self._tanh(x[0])
        x = self._flatten(x)
        x = self._linear(x)

        return x


# if __name__ == '__main__':
#     model = CnnLSTM(out_conv_filters=32, conv_kernel=1, conv_padding="same", pool_padding="same", pool_size=1,
#                     lstm_hidden_unit=64, n_features=5)
#
#     summary(model=model, input_data=(5, 10))
