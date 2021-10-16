from argparse import ArgumentParser, Namespace
from datetime import datetime
from core.dataloader import StockPriceDataset
from core.trainer import CnnLstmTrainer

from torch.utils.data import DataLoader
import torch

from settings import output_dir, has_cuda

from core.loss import LOSS_FACTORY
from core.utils import preprocessing
from core.dataloader import StockPriceDataset
from core.model import CnnLSTM

from pathlib import Path


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
                                         validation=False,
                                         col_name=arguments.col_name)

        validation_set = StockPriceDataset(train_size=arguments.train_size,
                                           filepath=arguments.in_file,
                                           test_size=arguments.test_size,
                                           phase="train",
                                           train=False,
                                           validation=True,
                                           time_step=arguments.time_step,
                                           save_plot=save_plot,
                                           col_name=arguments.col_name)

        test_set = StockPriceDataset(train_size=arguments.train_size,
                                     filepath=arguments.in_file,
                                     test_size=arguments.test_size,
                                     phase="train",
                                     time_step=arguments.time_step,
                                     train=False,
                                     validation=False,
                                     save_plot=save_plot,
                                     col_name=arguments.col_name)

        scale_conf = {
            "std": test_set.std_scale,
            "mean": test_set.mean_scale,
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
                                 n_features=arguments.n_features,
                                 lr=arguments.lr,
                                 loss=LOSS_FACTORY[arguments.loss],
                                 time_step=arguments.time_step)

        # train
        trainer.train(train_loader=train_loader,
                      epochs=arguments.epochs,
                      test_loader=test_loader,
                      save_path=run_dir,
                      scale=scale_conf,
                      validation_loader=validation_loader)
    elif arguments.preprocessing:
        i_file = Path(arguments.in_file)
        preprocessing(i_file)
    elif arguments.model != "":
        i_file = Path(arguments.in_file)
        test_ds = StockPriceDataset(filepath=str(i_file),
                                    time_step=arguments.time_step,
                                    train=False,
                                    validation=False,
                                    col_name=arguments.col_name,
                                    phase="test",
                                    save_plot=None)

        state_dic = torch.load(arguments.model)
        model = CnnLSTM(arguments.conv_filters, arguments.conv_kernel, arguments.conv_padding, arguments.pool_size,
                        arguments.pool_padding, arguments.lstm_hid,
                        arguments.n_features, time_step=arguments.time_step)
        if has_cuda:
            model.cuda()
        model.load_state_dict(state_dict=state_dic)

        last_day = test_ds[-1][0]
        last_day = torch.unsqueeze(last_day, dim=0)
        last_day = torch.transpose(last_day, dim0=1, dim1=2)

        std = test_ds.std_scale
        mean = test_ds.mean_scale

        if has_cuda:
            last_day = last_day.float().cuda()

        model.eval()
        with torch.no_grad():
            pred = model(last_day)
            pred = pred * std + mean

            print("Next Day Price: ", pred.cpu().numpy())



    else:
        print("[Failed] you had selected a wrong option")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train", help="enable training process", action="store_true")
    parser.add_argument("--validation", help="enable validation process", action="store_true")
    parser.add_argument("--input", dest="in_file", help="input file to train or validation", type=str, required=True)
    parser.add_argument("--model", help="saved model state dic for validation", type=str, default="")
    parser.add_argument("--col_name", help="label column name", type=str, default="")
    parser.add_argument("--preprocessing", help="process the given data", action="store_true")

    # hyper parameter
    # convolution
    parser.add_argument("--n_features", help="number of features", type=int, default=7)
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
