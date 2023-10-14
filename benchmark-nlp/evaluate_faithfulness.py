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
    experiment_id = 'eval-l'
    expl_id = args.exp_path.split('/')[-1][:-4]
    return f'{experiment_id}__{expl_id}'


def main(args):
    # get data
    dataset = get_data(args)
    print(args.dataset_name)
    print(len(dataset))

    # get model
    model, tokenizer = get_model(args)

    # get computed explanation results
    exp_results = dict(np.load(args.exp_path, allow_pickle=True).items())

    eval_configs = args.eval_configs
    eval_results = {}
    for i, raw_text in enumerate(tqdm(dataset, leave=True, position=0)):
        sample_str_id = str(i) + '-in-' + args.dataset_name

        pert = it.PerturbationNLP(model)
        expl = exp_results[sample_str_id].item()['exp']
        eval_result = {}
        if 'smoothgrad' in args.exp_path or 'intgrad' in args.exp_path:

            r = pert.evaluate(raw_text, expl['sq'], tokenizer, **eval_configs)
            tmp = copy.deepcopy(r)
            tmp.pop('LeRF_samples', None)
            tmp.pop('MoRF_samples', None)
            eval_result['sq'] = tmp

            r = pert.evaluate(raw_text, expl['abs'], tokenizer, **eval_configs)
            tmp = copy.deepcopy(r)
            tmp.pop('LeRF_samples', None)
            tmp.pop('MoRF_samples', None)
            eval_result['abs'] = tmp

            r = pert.evaluate(raw_text, expl['sum'], tokenizer, **eval_configs)
            tmp = copy.deepcopy(r)
            tmp.pop('LeRF_samples', None)
            tmp.pop('MoRF_samples', None)
            eval_result['sum'] = tmp

        elif 'lime' in args.exp_path:
            k = list(expl.keys())[0]
            lime_expl = expl[k]
            lime_expl_array = np.zeros(len(lime_expl))
            for i, v in lime_expl:
                lime_expl_array[i] = v

            r = pert.evaluate(raw_text, lime_expl_array, tokenizer, **eval_configs)
            tmp = copy.deepcopy(r)
            tmp.pop('LeRF_samples', None)
            tmp.pop('MoRF_samples', None)
            eval_result['oo'] = tmp
        else:
            r = pert.evaluate(raw_text, expl, tokenizer, **eval_configs)
            tmp = copy.deepcopy(r)
            tmp.pop('LeRF_samples', None)
            tmp.pop('MoRF_samples', None)
            eval_result['oo'] = tmp

        tmp = copy.deepcopy(eval_result)
        eval_results[sample_str_id] = tmp
        
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
