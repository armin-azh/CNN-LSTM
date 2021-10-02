from argparse import ArgumentParser, Namespace
from datetime import datetime
from core.dataloader import StockPriceDataset
from core.trainer import CnnLstmTrainer

from torch.utils.data import DataLoader

from settings import output_dir

from core.loss import LOSS_FACTORY


def main(arguments: Namespace):
    if arguments.train:

        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        run_dir = output_dir.joinpath(_cu)
        run_dir.mkdir(parents=True, exist_ok=True)

        save_plot = run_dir.joinpath("plot")
        save_plot.mkdir(parents=True, exist_ok=True)

        print(f"Training is starting ...")
        print(f"[Train] Loading the dataset")
        training_set = StockPriceDataset(train_size=arguments.train_size,
                                         filepath=arguments.in_file,
                                         test_size=arguments.test_size,
                                         phase="train",
                                         time_step=arguments.time_step,
                                         save_plot=save_plot,
                                         train=True,
                                         validation=False)

        validation_set = StockPriceDataset(train_size=arguments.train_size,
                                           filepath=arguments.in_file,
                                           test_size=arguments.test_size,
                                           phase="train",
                                           train=False,
                                           validation=True,
                                           time_step=arguments.time_step,
                                           save_plot=save_plot)

        test_set = StockPriceDataset(train_size=arguments.train_size,
                                     filepath=arguments.in_file,
                                     test_size=arguments.test_size,
                                     phase="train",
                                     time_step=arguments.time_step,
                                     train=False,
                                     validation=False,
                                     save_plot=save_plot)

        scale_conf = {
            "std": test_set.std_scale,
            "mean": test_set.mean_scale
        }
        print(
            f"[Train] Train: {len(training_set)} samples\tValidation: {len(validation_set)} samples\tTest: {len(test_set)} samples")

        train_loader = DataLoader(dataset=training_set,
                                  batch_size=arguments.batch_size,
                                  shuffle=False,
                                  num_workers=arguments.n_worker)

        validation_loader = DataLoader(dataset=validation_set,
                                       batch_size=arguments.batch_size,
                                       shuffle=False,
                                       num_workers=arguments.n_worker)

        test_loader = DataLoader(dataset=test_set,
                                 batch_size=len(test_set),
                                 shuffle=False,
                                 num_workers=arguments.n_worker)

        trainer = CnnLstmTrainer(out_conv_filters=arguments.conv_filters,
                                 conv_kernel=arguments.conv_kernel,
                                 conv_padding=arguments.conv_padding,
                                 pool_size=arguments.pool_size,
                                 pool_padding=arguments.pool_padding,
                                 lstm_hidden_unit=arguments.lstm_hid,
                                 n_features=5, lr=arguments.lr,
                                 loss=LOSS_FACTORY[arguments.loss])

        # train
        trainer.train(train_loader=train_loader,
                      epochs=arguments.epochs,
                      test_loader=test_loader,
                      save_path=run_dir,
                      scale=scale_conf,
                      validation_loader=validation_loader)
    elif arguments.validation:
        pass
    else:
        print("[Failed] you had selected a wrong option")


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
    parser.add_argument("--time_step", help="number of time step", type=int, default=10)

    # loss
    parser.add_argument("--loss", help="determine loss method", default="mae", type=str, choices=["rmse", "mae", "r"])

    # learning
    parser.add_argument("--batch_size", help="determine the batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=500)
    parser.add_argument("--train_size", help="training size percentage", type=float, default=0.9)
    parser.add_argument("--test_size", help="test size percentage", type=float, default=0.1)
    parser.add_argument("--n_worker", help="number of workers", type=int, default=4)

    args = parser.parse_args()

    main(arguments=args)
