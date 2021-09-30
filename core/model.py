from torch import nn
from torch import Tensor

from torchsummary import summary


class CnnLSTM(nn.Module):
    def __init__(self, out_conv_filters: int, conv_kernel: int, conv_padding: str, pool_size: int, pool_padding: str,
                 lstm_hidden_unit: int, n_features: int):
        super(CnnLSTM, self).__init__()

        self._net = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=out_conv_filters, kernel_size=conv_kernel,
                      padding=conv_padding),
            nn.Tanh(),

            nn.MaxPool1d(kernel_size=pool_size),
            nn.ReLU(inplace=True),

            nn.LSTM(batch_first=True, hidden_size=lstm_hidden_unit, input_size=10, num_layers=1, bidirectional=False),
            nn.Tanh(),

            nn.Linear(in_features=32 * 64, out_features=1)

        )

    def forward(self, x: Tensor) -> Tensor:
        return self._net(x)


# if __name__ == '__main__':
#     model = CnnLSTM(out_conv_filters=32, conv_kernel=1, conv_padding="same", pool_padding="same", pool_size=1,
#                     lstm_hidden_unit=64, n_features=5)
#
#     summary(model=model, input_data=(5, 10))
