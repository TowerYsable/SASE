from argparse import ArgumentParser

import numpy as np
import torch

import flwr as fl

import torch.optim as optim
import torch.utils.data

from model.conv_stft import ConvSTFT
from utils.losses import *
from my_network import *
import warnings


if __name__ == "__main__":
    # Training settings
    warnings.filterwarnings(action='ignore')
    parser = ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--server_address",
        type=str,
        # default="[::]:9999",
        default="localhost:8080",
        help=f"gRPC server address (default: '[::]:8080')",
    )
    parser.add_argument('--clean-train-dir', type=str, default="/root/tower/dataset/val_data/clean_trainset_56spk_wav")
    parser.add_argument('--noisy-train-dir', type=str, default="/root/tower/dataset/val_data/noisy_trainset_56spk_wav")
    parser.add_argument('--clean-valid-dir', type=str, default="/root/tower/dataset/val_data/clean_dev")
    parser.add_argument('--noisy-valid-dir', type=str, default="/root/tower/dataset/val_data/noise_dev")
    parser.add_argument('--clean-test-dir', type=str, default="/root/tower/dataset/val_data/clean_testset_wav")
    parser.add_argument('--noisy-test-dir', type=str, default="/root/tower/dataset/val_data/noisy_testset_wav")

    parser.add_argument('--epochs', type=int, default=50, help='Number of max epochs in training (default: 100)')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=16, help='Number of workers in dataset loader (default: 4)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size in training (default: 32)')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--lr-decay', default=0.1)
    parser.add_argument('--weight-decay', default=1e-5)
    parser.add_argument('--arch', type=str, default="CL", help='模式选择')
    parser.add_argument('--sample-rate', type=int, default=48000, help="STFT hyperparam")
    parser.add_argument('--max-len', type=int, default=165000)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--resume', default="/root/tower/speech_enhancement_exper/Speech_Enhancement-DCCRN/fed_saved_models/model_1.pth", type=str, metavar='PATH', help="model_args.resume")
    parser.add_argument('--evaluate', '-e', default=False, action='store_true')
    # generate
    parser.add_argument('--generate', '-g', default=False, action='store_true')
    parser.add_argument('--denoising-file', '-df',
                        default="/root/linzhentao/experiment/denoiser/dataset/valentini/noisy_testset_wav",
                        type=str, help="想要去噪的文件路径")

    parser.add_argument('--loss-mode', '-lm', type=str, default="SI-SNR", help="loss")

    args = parser.parse_args()

    # Load MNIST data
    train_loader, test_loader = load_data(args)

    # pylint: disable=no-member
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pylint: enable=no-member
    model = set_model(args, mode='CL')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    # model.to(device)
    # Instantiate client
    client = NetClient(model,train_loader,test_loader)

    # Start client
    fl.client.start_client("localhost:8080", client)
