
from matplotlib.pyplot import axis
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
from sklearn.metrics.pairwise import cosine_similarity

from args import args


def get_exp_id(args):
    data_id = args.data_list.split('/')[-1][:-4]
    
    if args.name == 'default':
        it = f'{args.it}'
    else:
        it = f'{args.name}'

    return f'eval-consensus_{it}_{data_id}'

def parsing_path(path):
    model_names = ['mae_vit', 'vit_small', 'vit_base', 'vit_large',
                  "ResNet50", "ResNet101", "ResNet152", "mobilenet", "vgg16"]
    
    for model_name in model_names:
        if model_name in path:
            model = model_name
            break
    
    return model

def load_results(it_name):
    path_list = glob(f'work_dirs/*{it_name}*.npz')
    path_list = [p for p in path_list if 'eval' not in p]
    
    all_models_results_dict = {}
    for p in path_list:
        tmp = dict(np.load(p, allow_pickle=True).items())
        all_models_results_dict[p] = tmp
    
    model_name_list = [parsing_path(p) for p in list(all_models_results_dict.keys())]

    return all_models_results_dict, tmp.keys(), model_name_list

def main(args):
    # because we have computed the explanations and saved them,
    # there is no need to recompute them.

    # get it name
    it_name = args.it

    # find all related results
    all_models_results_dict, img_path_list, model_name_list = load_results(it_name)

    cos_scores_ndarray = np.zeros((len(img_path_list), len(model_name_list)))
    for i, img_path in enumerate(img_path_list):
        # collect results
        all_expl_list = []
        for k in all_models_results_dict:
            expl = all_models_results_dict[k][img_path].item()['exp']
            expl = expl.astype(np.float32)
            all_expl_list.append(expl)
        all_expl = np.concatenate(all_expl_list, axis=0)

        # compute consensus
        if len(all_expl.shape) == 4:
            all_expl = np.square(all_expl).mean(axis=1)
        consensus = all_expl.mean(axis=0, keepdims=True)  # [1, 224, 224] or [1, 14, 14] or [1, 7, 7]

        # compute consensus score
        cos_scores = cosine_similarity(consensus.reshape(1, -1), all_expl.reshape(all_expl.shape[0], -1))
        
        cos_scores_ndarray[i] = cos_scores
    
    cos_scores_final = np.mean(cos_scores_ndarray, axis=0)
    eval_results = {model_name: cos_scores_final[i] for i, model_name in enumerate(model_name_list)}

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
