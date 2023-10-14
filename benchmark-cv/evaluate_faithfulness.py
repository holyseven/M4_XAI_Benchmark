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
    experiment_id = 'eval'
    if args.aggregate == 'square':
        experiment_id += '-sq'

    data_id = args.data_list.split('/')[-1][:-4]
    expl_id = args.exp_path.split('/')[-1][:-4]

    if args.name not in expl_id:
        experiment_id += f'-{args.name}'
    
    if data_id not in expl_id:
        experiment_id += f'_{expl_id}_{data_id}'
    else:
        experiment_id += f'_{expl_id}'

    return experiment_id


def main(args):
    # get data
    list_image_paths = get_data(args)
    print(args.data_list)
    print(len(list_image_paths))

    # get model
    paddle_model = get_model(args)

    # eval configs
    eval_configs = args.eval_configs

    # get computed explanation results
    exp_results = dict(np.load(args.exp_path, allow_pickle=True).items())

    img_resize_configs = args.img_resize_configs
    if img_resize_configs is None:
        img_resize_configs = {"resize_to": 224, "crop_to": 224}

    eval_results = {}
    for i, img_path in enumerate(tqdm(list_image_paths, leave=True, position=0)):
        expl = exp_results[img_path].item()['exp']
        expl = expl.astype(np.float32)

        if len(expl.shape) == 4:
            if args.aggregate == 'abs':
                expl = np.abs(expl).mean(axis=(0,1))
            elif args.aggregate == 'square':
                expl = np.square(expl).mean(axis=(0,1))
            else:
                raise ValueError(f'Unknown args.aggregate: {args.aggregate}.')

        elif len(expl.shape) == 3 and expl.shape[0] == 1:
            expl = expl[0]
        else:
            raise ValueError("Unknow explanation type.")

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
