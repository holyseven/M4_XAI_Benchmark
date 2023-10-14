
import paddle
import interpretdl as it

import os
import copy
import logging
import os.path as osp
import numpy as np
from tqdm import tqdm
from datetime import datetime

from paddle.io import Dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from args import args
from multihead import layer_replacement

import sys
sys.path.insert(0, '..')
from training_nlp.data import SST2, MovieReview

model_names = ["ernie-2.0-base", "bert-base-uncased", "bert-large-uncased", "roberta-base", "distilbert-base-uncased"]

def get_exp_id(args):
    if args.it == 'ga':
        it_configs = args.it_configs
        it = args.it + '-' + str(it_configs['start_layer'])
    elif args.it == 'bt':
        it_configs = args.it_configs
        it = args.it + '-' + it_configs['ap_mode'] + '-' + str(it_configs['start_layer'])
    else:
        it = args.it

    return f'{args.name}__{args.model_name}__{it}__{args.dataset_name}'


def get_data(args):
    if args.dataset_name == 'sst-2':
        lines = SST2(args.dataset_root, split_name='dev').lines
        dataset = [l.strip().split(' \t')[0] for l in lines]
    elif args.dataset_name == 'movies':
        dataset = MovieReview(args.dataset_root, split_name='val').samples
    else:
        pass

    return dataset
    
def get_model(args):
    model_name = args.model_name
    dataset_name = args.dataset_name

    model_weights = f"{args.model_root}/{model_name}_{dataset_name}/model.pd"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load trained weights
    print('Loading weights from', model_weights)
    state_dict = paddle.load(model_weights)
    model.set_dict(state_dict)
    
    # msa layer.
    model = layer_replacement(model)

    return model, tokenizer

def get_it(args, paddle_model):
    interpreters_dict = {
        'lime': it.LIMENLPInterpreter,
        'intgrad': it.IntGradNLPInterpreter,
        'smoothgrad': it.SmoothGradNLPInterpreter,
        'ga': it.GANLPInterpreter,
        'bt': it.BTNLPInterpreter
    }

    return interpreters_dict[args.it](paddle_model, device=args.device)

def main(args):

    # get data
    dataset = get_data(args)
    print(args.dataset_name)
    print(len(dataset))

    # get model
    model, tokenizer = get_model(args)

    # get interpreter instance
    interpreter = get_it(args, model)
    
    # interpreter configs
    it_configs = args.it_configs

    exp_results = {}
    if os.path.exists(f'./work_dirs/{get_exp_id(args)}.npz'):
        logging.info(f"Loading computed results from ./work_dirs/{get_exp_id(args)}.npz")
        exp_results = dict(np.load(f'./work_dirs/{get_exp_id(args)}.npz', allow_pickle=True))
    
    for i, raw_text in enumerate(tqdm(dataset, leave=True, position=0)):
        sample_str_id = str(i) + '-in-' + args.dataset_name
        if sample_str_id in exp_results:
            # load computed exp.
            continue
            exp_result = exp_results[img_path].item()
        else:
            # compute exp. lime_results or array_exp.
            exp = interpreter.interpret(raw_text, tokenizer, **it_configs, visual=False)

            if args.it in ['smoothgrad', 'intgrad']:
                exp = {
                    'sq': np.sum(np.square(exp), axis=-1), 
                    'abs': np.sum(np.abs(exp), axis=-1),
                    'sum': np.sum(exp, axis=-1)
                }
            
            exp_result = {'exp': exp, 'proba': interpreter.predicted_proba}
            exp_results[sample_str_id] = copy.deepcopy(exp_result)
        
        if args.save_eval_result and i % (len(dataset)//10) == 0:
            np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **exp_results)
            logging.info(f'Saving exp_results at {i}')
    
    if args.save_eval_result:
        np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **exp_results)
        logging.info(f'Saving exp_results at the end.')

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
