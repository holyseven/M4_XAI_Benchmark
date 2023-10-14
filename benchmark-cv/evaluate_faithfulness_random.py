import paddle
import interpretdl as it

import os
import copy
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

from args import args
from interpret_and_save import get_data, get_model


def get_exp_id(args):
    data_id = args.data_list.split('/')[-1][:-4]
    model_id = args.model
    expl_id = args.exp_path
    return f'eval-{args.name}_{model_id}_{expl_id}_{data_id}'

def main(args):
    # get data
    list_image_paths = get_data(args)
    print(args.data_list)
    print(len(list_image_paths))

    # get model
    paddle_model = get_model(args)

    # eval configs
    eval_configs = args.eval_configs

    random_7 = {}
    random_14 = {}
    random_224 = {}
    constant_14 = {}
    constant_14_each = np.ones((14, 14))
    constant_14_each[2:12, 2:12] = 1
    for img_path in list_image_paths:
        random_224[img_path] = np.random.rand(224, 224)
        random_14[img_path] = np.random.rand(14, 14)
        random_7[img_path] = np.random.rand(7, 7)
        constant_14[img_path] = constant_14_each

    # get computed explanation results
    if args.exp_path == 'random_7':
        exp_results = random_7
    elif args.exp_path == 'random_14':
        exp_results = random_14
    elif args.exp_path == 'random_224':
        exp_results = random_224
    elif args.exp_path == 'constant_14':
        exp_results = constant_14
    else:
        raise ValueError(f"unknown {args.exp_path}. ")

    img_resize_configs = args.img_resize_configs
    if img_resize_configs is None:
        img_resize_configs = {"resize_to": 224, "crop_to": 224}

    eval_results = {}
    for i, img_path in enumerate(tqdm(list_image_paths, leave=True, position=0)):
        expl = exp_results[img_path].astype(np.float32)

        perturbation = it.Perturbation(paddle_model, 'gpu:0')
        eval_result = perturbation.evaluate(img_path, expl, **eval_configs, **img_resize_configs)
        tmp = copy.deepcopy(eval_result)
        tmp.pop('LeRF_images', None)
        tmp.pop('MoRF_images', None)
        eval_results[img_path] = tmp
        
        # if args.save_eval_result and i % (len(list_image_paths)//10) == 0:
        #     np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **eval_results)
        #     logging.info(f'Saving eval_results at {i}')
    
    if args.save_eval_result:
        np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **eval_results)
        logging.info(f'Saving eval_results at the end.')


if __name__ == '__main__':
    assert args.exp_path in ['random_7', 'random_14', 'random_224', 'constant_14']

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
