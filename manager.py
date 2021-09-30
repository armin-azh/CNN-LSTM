from argparse import ArgumentParser, Namespace


def main(arguments: Namespace):
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train", help="enable training process", action="store_true")
    parser.add_argument("--validation", help="enable validation process", action="store_true")
    parser.add_argument("--input", dest="in_file", help="input file to train or validation", type=str, required=True)
    parser.add_argument("--model", help="saved model state dic for validation", type=str, default="")

    # hyper parameter
    # convolution
    parser.add_argument("--conv_filters", help="number of convolution filters", type=int, default=32)
    parser.add_argument("--conv_kernel", help="convolution kernel size dimension", type=int, default=1)
    parser.add_argument("--conv_padding", help="convolution padding type", type=str, default="same",
                        choices=["valid", "same"])

    # pooling
    parser.add_argument("--pool_size", help="pool size", type=str, default=1)
    parser.add_argument("--pool_padding", help="pool padding type", type=str, default="same", choices=["valid", "same"])

    # lstm
    parser.add_argument("--lstm_hid", help="number of lstm hidden unit", type=int, default=64)
    parser.add_argument("-time_step", help="number of time step", type=int, default=10)

    # learning
    parser.add_argument("--batch_size", help="determine the batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=100)

    args = parser.parse_args()

    main(arguments=args)
