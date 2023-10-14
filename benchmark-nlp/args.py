import argparse
import json

parser = argparse.ArgumentParser()
# data
parser.add_argument('--dataset_name', default="sst-2", type=str, help="data_list")
parser.add_argument('--dataset_root', default="/root/.paddlenlp/datasets/Glue/SST-2", type=str, help="data_list")

# exp id prefix
parser.add_argument('--name', default="default", type=str, help="name of experiments")
# model
parser.add_argument('--model_name', default="ernie-2.0-base-en", type=str, help="model name")
parser.add_argument('--model_root', default="/root/codespace/benchmark-it/training_nlp/results", type=str, help="model name")

# it
parser.add_argument('--it', default="lime", type=str, help="interpreter name")
parser.add_argument('--it_configs', default='{}', type=json.loads, help="arguments for interpreter")

# save
parser.add_argument('--save_eval_result', default=0, type=int, help="save explanations")

# eval
parser.add_argument('--eval_configs', default='{}', type=json.loads, help="arguments for evaluator")
parser.add_argument('--device', default="gpu:0", type=str, help="device")
parser.add_argument('--eval_num_limit_adapter', default=None, type=str, help="arguments for evaluator")
parser.add_argument('--exp_path', default=None, type=str, help="exp_path")
parser.add_argument('--aggregate', default='abs', type=str, help="abs or square for input-gradient based methods")

# used for glime only.
parser.add_argument('--global_weights', default=None, type=str, help="./work_dirs/global_weights_normlime.npy")
args = parser.parse_args()
