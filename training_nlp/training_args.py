import argparse
import json

parser = argparse.ArgumentParser()
# exp id prefix
parser.add_argument('--name', default="default", type=str, help="name of experiments")

# data
parser.add_argument('--dataset_root', default="/root/.paddlenlp/datasets/Glue/SST-2", type=str, help="dataset root")
parser.add_argument('--batch_size', default=64, type=int, help="dataset root")
parser.add_argument('--batch_size_eval', default=100, type=int, help="dataset root")
parser.add_argument('--max_seq_length', default=128, type=int, help="ckpt num")

# model
parser.add_argument('--model_name', default="ernie-2.0-base-en", type=str, help="model name")
parser.add_argument('--ckpt_num', default=5, type=int, help="ckpt num")
parser.add_argument('--model_weights', default=None, type=str, help="trained model path")

# optmization
parser.add_argument('--lr', default=5e-5, type=float, help="model name")
parser.add_argument('--epochs', default=1, type=int, help="model name")

# logging
parser.add_argument('--log', default="INFO", help="Logging level.")

# random seed
parser.add_argument('--random_seed', default=77, type=int, help="")

# env
parser.add_argument('--device', default="gpu:0", type=str, help="device")

args = parser.parse_args()
