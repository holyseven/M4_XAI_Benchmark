import paddle
import interpretdl as it

import os
import gc
import copy
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
from scipy.stats import entropy

from args import args
from interpret_and_save import get_data, get_model


def get_exp_id(args, expl_id):
    experiment_id = f'eval2'
    if args.name not in expl_id:
        experiment_id += f'-{args.name}'

    data_id = args.data_list.split('/')[-1][:-4]
    if data_id not in expl_id:
        experiment_id += f'_{expl_id}_{data_id}'
    else:
        experiment_id += f'_{expl_id}'

    return experiment_id


def sparsity_v1(expl):
    if expl.max() == expl.min():
        expl += 1.0
        return entropy(expl, base=2, axis=None)

    expl /= np.sum(expl)
    return entropy(expl, base=2, axis=None)


def main(args):
    # get data, for reading paths only
    list_image_paths = get_data(args)
    # print(args.data_list)
    print(len(list_image_paths))

    # get model
    paddle_model = get_model(args)

    # eval configs
    eval_configs = args.eval_configs

    img_resize_configs = args.img_resize_configs
    if img_resize_configs is None:
        img_resize_configs = {"resize_to": 224, "crop_to": 224}

    # get computed explanation results
    if 'mae_vit' == args.model:
        all_expl_paths = sorted([p for p in glob('work_dirs/*mae_vit*.npz') if 'eval' not in p and 'lime' not in p])
    else:
        all_expl_paths = sorted([p for p in glob(f'work_dirs/*{args.model}*.npz') if 'eval' not in p and 'lime' not in p and 'mae_vit' not in p])

    print(all_expl_paths)

    expl_results = {}
    for exp_path in all_expl_paths:
        expl_id = exp_path.split('/')[-1][:-4]
        exp_result = dict(np.load(exp_path, allow_pickle=True).items())
        if 'smoothgrad' in expl_id or 'intgrad' in expl_id:
            expl_results[expl_id + '-abs'] = exp_result
            expl_results[expl_id + '-square'] = exp_result
        else:
            expl_results[expl_id] = exp_result

    random_7 = {}
    random_14 = {}
    random_224 = {}
    constant_14 = {}
    constant_14_each = np.ones((1, 14, 14))
    constant_14_each[0, 2:12, 2:12] = 1
    for img_path in exp_result.keys():
        random_224[img_path] = np.random.rand(1, 224, 224)
        random_14[img_path] = np.random.rand(1, 14, 14)
        random_7[img_path] = np.random.rand(1, 7, 7)
        constant_14[img_path] = constant_14_each
        
    expl_results[args.model + '_random_7'] = random_7
    expl_results[args.model + '_random_14'] = random_14
    expl_results[args.model + '_random_224'] = random_224
    expl_results[args.model + '_constant_14'] = constant_14
    
    eval_results_dict = {}
    # eval_results_dict: 
    # {
    # expl_id: 
    #   {
    #       img_path: {'sparsity': sparsity, 'infidelity': infi_score}
    #   }
    # }
    for i, img_path in tqdm(enumerate(exp_result.keys())):
        infi_evalutor = it.Infidelity(paddle_model)

        for expl_id in expl_results:
            if expl_id in [
                    args.model + '_random_7',
                    args.model + '_random_14',
                    args.model + '_random_224',
                    args.model + '_constant_14']:
                expl = expl_results[expl_id][img_path]
            else:
                expl = expl_results[expl_id][img_path].item()['exp']
            
            expl = expl.astype(np.float32)
            if len(expl.shape) == 4:
                if 'abs' in expl_id:
                    expl = np.abs(expl).mean(axis=(0,1))
                elif 'square' in expl_id:
                    expl = np.square(expl).mean(axis=(0,1))
                else:
                    raise ValueError(f'Unexpected {expl_id}.')
            elif len(expl.shape) == 3 and expl.shape[0] == 1:
                expl = expl[0]
            else:
                raise ValueError("Unknow explanation type.")

            # infidelity
            infi_score = infi_evalutor.evaluate(img_path, expl, **eval_configs, **img_resize_configs)
            
            # sparsity
            sparsity = sparsity_v1(expl)
            tmp = {'sparsity': sparsity, 'infidelity': infi_score}
            
            eval_results = eval_results_dict.get(expl_id, {})
            eval_results[img_path] = tmp
            eval_results_dict[expl_id] = eval_results

        del infi_evalutor
        gc.collect()

        if args.save_eval_result and i % (len(list_image_paths)//10) == 0:
            for expl_id in eval_results_dict:
                np.savez(f'./work_dirs/{get_exp_id(args, expl_id)}.npz', **eval_results_dict[expl_id])
                logging.info(f'Saving eval_results at {i}')

    for expl_id in eval_results_dict:
        np.savez(f'./work_dirs/{get_exp_id(args, expl_id)}.npz', **eval_results_dict[expl_id])
        logging.info(f'Saving eval_results at the end.')


if __name__ == '__main__':

    tik = datetime.now()
    os.makedirs('./work_dirs', exist_ok=True)
    FORMAT = '%(asctime)-15s %(message)s'
    
    logging.basicConfig(
        filename=f"./work_dirs/{get_exp_id(args, args.model + '-all-exp')}.log", 
        # filemode='w', 
        format=FORMAT,
        level=getattr(logging, 'INFO')
    )
    logging.info(f'{args}\n')
    print(args)

    main(args)

    logging.info(f"Time: {datetime.now() - tik} s.")
