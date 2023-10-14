
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
    experiment_id = 'eval2'
    return f'{experiment_id}__{args.model_name}__{args.dataset_name}__{expl_id}'


def sparsity_v1(expl):
    expl = np.squeeze(expl)
    expl = expl[1:-1]
    if expl.max() == expl.min():
        expl += 1.0
        return entropy(expl, base=2, axis=None)

    expl /= np.sum(expl)
    return entropy(expl, base=2, axis=None)


def main(args):
    # get data
    dataset = get_data(args)
    print('dataset name:', args.dataset_name)
    print('dataset numple of samples:', len(dataset))

    # get model
    model, tokenizer = get_model(args)

    # get computed explanation results, except LIME.
    all_expl_paths = sorted([p for p in glob(f'work_dirs/*__{args.model_name}__*.npz') if 'eval' not in p])
    print(all_expl_paths, len(all_expl_paths))

    expl_results = {}
    for exp_path in all_expl_paths:
        prefix_expl_id = exp_path.split('/')[-1][:-4]
        exp_result = dict(np.load(exp_path, allow_pickle=True).items())
        expl_results[prefix_expl_id] = exp_result

    eval_configs = args.eval_configs
    eval_results_dict = {}
    # eval_results_dict: 
    # {
    # expl_id: 
    #   {
    #       img_path: {'sparsity': sparsity, 'infidelity': infi_score}
    #   }
    # }
    for text_index, raw_text in enumerate(tqdm(dataset, leave=True, position=0)):
        sample_str_id = str(text_index) + '-in-' + args.dataset_name

        infi_evalutor = it.InfidelityNLP(model)
        for prefix_expl_id in expl_results:
            exp_result = expl_results[prefix_expl_id]
            expl = exp_result[sample_str_id].item()['exp']

            if 'smoothgrad' in prefix_expl_id or 'intgrad' in prefix_expl_id:
                # sq
                expl_id = prefix_expl_id + '-sq'
                infi_score = infi_evalutor.evaluate(raw_text, expl['sq'], tokenizer, **eval_configs)
                sparsity = sparsity_v1(expl['sq'])

                tmp = {'sparsity': sparsity, 'infidelity': infi_score}
                eval_results = eval_results_dict.get(expl_id, {})
                eval_results[sample_str_id] = tmp
                eval_results_dict[expl_id] = eval_results

                # abs
                expl_id = prefix_expl_id + '-abs'
                infi_score = infi_evalutor.evaluate(raw_text, expl['abs'], tokenizer, **eval_configs)
                sparsity = sparsity_v1(expl['abs'])

                tmp = {'sparsity': sparsity, 'infidelity': infi_score}
                eval_results = eval_results_dict.get(expl_id, {})
                eval_results[sample_str_id] = tmp
                eval_results_dict[expl_id] = eval_results

                # sum
                expl_id = prefix_expl_id + '-sum'
                infi_score = infi_evalutor.evaluate(raw_text, expl['sum'], tokenizer, **eval_configs)
                sparsity = sparsity_v1(expl['sum'])

                tmp = {'sparsity': sparsity, 'infidelity': infi_score}
                eval_results = eval_results_dict.get(expl_id, {})
                eval_results[sample_str_id] = tmp
                eval_results_dict[expl_id] = eval_results

                expl_shape = expl['abs'].shape
                
            elif 'lime' in prefix_expl_id:
                k = list(expl.keys())[0]
                lime_expl = expl[k]
                lime_expl_array = np.zeros(len(lime_expl))
                for i, v in lime_expl:
                    lime_expl_array[i] = v
                infi_score = infi_evalutor.evaluate(raw_text, lime_expl_array, tokenizer, **eval_configs)
                sparsity = sparsity_v1(np.abs(lime_expl_array))

                expl_id = prefix_expl_id
                tmp = {'sparsity': sparsity, 'infidelity': infi_score}
                eval_results = eval_results_dict.get(expl_id, {})
                eval_results[sample_str_id] = tmp
                eval_results_dict[expl_id] = eval_results

                expl_shape = lime_expl_array.shape
            else:
                infi_score = infi_evalutor.evaluate(raw_text, expl, tokenizer, **eval_configs)
                sparsity = sparsity_v1(expl)

                expl_id = prefix_expl_id
                tmp = {'sparsity': sparsity, 'infidelity': infi_score}
                eval_results = eval_results_dict.get(expl_id, {})
                eval_results[sample_str_id] = tmp
                eval_results_dict[expl_id] = eval_results

                expl_shape = expl

        # random
        expl_id = args.model_name + '__random__' + args.dataset_name
        expl = np.random.rand(*expl_shape)
        infi_score = infi_evalutor.evaluate(raw_text, expl, tokenizer, **eval_configs)
        sparsity = sparsity_v1(expl)
        tmp = {'sparsity': sparsity, 'infidelity': infi_score}
        eval_results = eval_results_dict.get(expl_id, {})
        eval_results[sample_str_id] = tmp
        eval_results_dict[expl_id] = eval_results

        # constant
        expl_id = args.model_name + '__constant__' + args.dataset_name
        expl = np.zeros(expl_shape)
        infi_score = infi_evalutor.evaluate(raw_text, expl, tokenizer, **eval_configs)
        sparsity = sparsity_v1(expl)
        tmp = {'sparsity': sparsity, 'infidelity': infi_score}
        eval_results = eval_results_dict.get(expl_id, {})
        eval_results[sample_str_id] = tmp
        eval_results_dict[expl_id] = eval_results

        del infi_evalutor
        gc.collect()
    
    np.savez(f"./work_dirs/{get_exp_id(args, 'all')}.npz", **eval_results_dict)
    logging.info(f'Saving eval_results at the end.')


if __name__ == '__main__':

    tik = datetime.now()
    os.makedirs('./work_dirs', exist_ok=True)
    FORMAT = '%(asctime)-15s %(message)s'

    logging.basicConfig(
        filename=f"./work_dirs/{get_exp_id(args, 'all-exp')}.log", 
        # filemode='w', 
        format=FORMAT,
        level=getattr(logging, 'INFO')
    )
    logging.info(f'{args}\n')
    print(args)

    main(args)

    logging.info(f"Time: {datetime.now() - tik} s.")
