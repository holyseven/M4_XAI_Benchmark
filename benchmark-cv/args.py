import argparse
import json

parser = argparse.ArgumentParser()
# data
parser.add_argument('--data_prefix', default="/root/codespace/benchmark-it/benchmark_data/imagenet_val_5k", type=str, help="data_list")
parser.add_argument('--data_list', default="/root/codespace/benchmark-it/benchmark_data/imagenet_val_5k.txt", type=str, help="data_list")
# parser.add_argument('--num_images', default=50, type=int, help="number of images for evaluation")

## data generation
# from glob import glob
# val_filesnames = sorted(glob('/root/datasets/ImageNet_org/val/*/*'))
# f = open('a.txt','w')
# for s in range(1):
#     for i in range(1000):
#         f.write(val_filesnames[i*50 + s].split('/root/datasets/ImageNet_org/val/')[-1] + '\n')
# f.flush()
# import os
# import shutil
# for s in range(5):
#     for i in range(1000):
#         src = val_filesnames[i*50 + s]
#         dst = src.split('/root/datasets/ImageNet_org/')[-1]
#         os.makedirs(os.path.dirname(dst), exist_ok=True)
#         shutil.copyfile(src, dst)

# exp id prefix
parser.add_argument('--name', default="default", type=str, help="name of experiments")
# model
parser.add_argument('--model', default="ResNet50", type=str, help="model name")
parser.add_argument('--model_weights', default=None, type=str, help="trained model path")
parser.add_argument('--num_classes', default=1000, type=int, help="number of classes")
# it
parser.add_argument('--it', default="lime", type=str, help="interpreter name")
parser.add_argument('--it_configs', default='{}', type=json.loads, help="arguments for interpreter")
parser.add_argument('--img_resize_configs', default=None, type=json.loads, help="arguments for evaluator")
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
