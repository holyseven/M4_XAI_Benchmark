import argparse
import json

parser = argparse.ArgumentParser()
# exp id prefix
parser.add_argument('--name', default="default", type=str, help="name of experiments")

# data
parser.add_argument('--dataset_root', default="/root/codespace/benchmark_data/imagenet_val_5k", type=str, help="dataset root")
parser.add_argument('--data_list', default="/root/codespace/benchmark_data/imagenet_val_5k.txt", type=str, help="data_list")
parser.add_argument('--val_dataset_root', default="/root/codespace/benchmark_data/imagenet_val_5k", type=str, help="dataset root")
parser.add_argument('--val_data_list', default="/root/codespace/benchmark_data/imagenet_val_5k.txt", type=str, help="val_data_list")
parser.add_argument('--resize_shorter_edge', default=256, type=int, help="dataset root")
parser.add_argument('--crop_size', default=224, type=int, help="dataset root")
parser.add_argument('--batch_size', default=64, type=int, help="dataset root")
parser.add_argument('--batch_size_eval', default=50, type=int, help="dataset root")

# model
parser.add_argument('--model_name', default="resnet50", type=str, help="model name")
parser.add_argument('--ckpt_num', default=5, type=int, help="ckpt num")
parser.add_argument('--num_classes', default=2, type=int, help="number of classes")
parser.add_argument('--model_weights', default=None, type=str, help="trained model path")

# optmization
parser.add_argument('--optimizer', default="Momentum", type=str, help="model name")
parser.add_argument('--lr', default=0.01, type=float, help="model name")
parser.add_argument('--epochs', default=50, type=int, help="model name")

# logging
parser.add_argument('--log', default="INFO", help="Logging level.")

# random seed
parser.add_argument('--random_seed', default=77, type=int, help="")

# env
parser.add_argument('--device', default="gpu:0", type=str, help="device")

args = parser.parse_args()
