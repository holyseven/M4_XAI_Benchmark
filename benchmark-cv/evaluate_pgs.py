
import paddle
import interpretdl as it

import os
import cv2
import copy
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from glob import glob
from PIL import Image

from paddle.vision import transforms
from paddle.io import Dataset

from args import args
from interpret_and_save import get_data, get_model, get_it


_image_backend = 'pil'

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def cv2_loader(path):
    img = cv2.imread(path)
    return img[:, :, ::-1]

def default_loader(path):
    if _image_backend == 'cv2':
        return cv2_loader(path)
    else:
        return pil_loader(path)

class Yellow(Dataset):
    def __init__(self,
                 dataroot,
                 data_list_file,
                 square_size=60,
                 mode='random',
                 transform=None):
        super(Dataset, self).__init__()

        self.dataroot = dataroot        
        self.transform = transform
        self.square_size = square_size
        self.mode = mode
        assert mode in ['random', 'val', 'validation']

        self.image_loader = default_loader
        self.samples = []

        # don't care about original labels
        with open(data_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if dataroot is not None:
                    each_line = os.path.join(dataroot, line.strip()) 
                else:
                    each_line = line.strip()
                self.samples.append(each_line)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        sample = self.image_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.mode == 'random':
            if np.random.rand() > 0.5:
                target = 1
                h, w = (np.random.rand(2) * (sample.shape[1] - self.square_size)).astype(np.int32)
                
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[0, h:(h+self.square_size), w:(w+self.square_size)] = 2.2489
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[1, h:(h+self.square_size), w:(w+self.square_size)] = 2.4286
                # (0.0 - 0.406) / 0.225 = -1.8044
                sample[2, h:(h+self.square_size), w:(w+self.square_size)] = -1.8044
            else:
                target = 0
        else:
            if idx < 2500:
                target = 1
                # 2500 validation
                _idx = idx

                choice_range = sample.shape[1] - self.square_size
                stride = choice_range / 50

                h = int((_idx // 50) * stride)
                w = int((_idx % 50) * stride)

                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[0, h:(h+self.square_size), w:(w+self.square_size)] = 2.2489
                # (1.0 - 0.485) / 0.229 = 2.2489
                sample[1, h:(h+self.square_size), w:(w+self.square_size)] = 2.4286
                # (0.0 - 0.406) / 0.225 = -1.8044
                sample[2, h:(h+self.square_size), w:(w+self.square_size)] = -1.8044
            else:
                target = 0

        return path, sample, h, w

def get_exp_id(args):
    data_id = args.data_list.split('/')[-1][:-4]
    
    if args.name == 'default':
        it = f'{args.it}'
    else:
        it = f'{args.name}'

    return f'eval-pgs_{args.model}_{it}_{data_id}'

def get_transform(crop_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    
    transform_val = transforms.Compose(
        [
            transforms.Resize(size=crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    return transform_val

def main(args):
    # get data
    yellow_dataset = Yellow(args.data_prefix, args.data_list, mode='val', transform=get_transform())

    # get model
    paddle_model = get_model(args)

    # get interpreter instance
    interpreter = get_it(args, paddle_model)
    if 'glime' == args.it:
        interpreter.set_global_weights(args.global_weights)
    
    # interpreter configs
    it_configs = args.it_configs

    # image resize config.
    # depreciated set: {"resize_to": 256, "crop_to": 224}
    img_resize_configs = args.img_resize_configs
    if img_resize_configs is None:
        img_resize_configs = {"resize_to": 224, "crop_to": 224}
    
    crop_to = img_resize_configs["crop_to"]
    eval_results = {}
    pgs = it.PointGameSegmentation()
    for i in tqdm(range(2500), leave=True, position=0):
        img_path, sample, h, w = yellow_dataset.__getitem__(i)

        # compute exp. lime_results or array_exp.
        expl = interpreter.interpret(np.expand_dims(sample.numpy(), axis=0), **it_configs, **img_resize_configs, visual=False)

        # compte gt.
        gt = np.zeros((crop_to, crop_to), np.uint8)
        gt[h:(h+yellow_dataset.square_size), w:(w+yellow_dataset.square_size)] = 1

        eval_result = {}

        if len(expl.shape) == 4:
            if 'sq' in args.name:
                expl = np.square(expl).mean(axis=(0,1))
            else:
                expl = np.abs(expl).mean(axis=(0,1))
        elif len(expl.shape) == 3 and expl.shape[0] == 1:
            expl = expl[0]
            expl = cv2.resize(expl, (crop_to, crop_to))
        else:
            raise ValueError("Unknow explanation type.")

        eval_result = pgs.evaluate(gt, expl)
        eval_results[img_path] = copy.deepcopy(eval_result)
    
    if args.save_eval_result:
        np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **eval_results)
        logging.info(f'Saving exp_results at the end.')
    
    return


if __name__ == '__main__':

    tik = datetime.now()
    os.makedirs('./work_dirs', exist_ok=True)
    FORMAT = '%(asctime)-15s %(message)s'
    
    logging.basicConfig(
        filename=f'./work_dirs/{get_exp_id(args)}.log', 
        # filemode='w', 
        format=FORMAT,
        level=getattr(logging, 'INFO')
    )
    logging.info(f'{args}\n')
    print(args)

    main(args)

    logging.info(f"Time: {datetime.now() - tik} s.")
