
import paddle
import interpretdl as it

import os
import copy
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

from args import args

def get_exp_id(args):
    data_id = args.data_list.split('/')[-1][:-4]
    return f'{args.name}_{args.model}_{args.it}_{data_id}'

def get_data(args):
    """Only support a txt file for providing image paths.
    """
    if '.txt' in args.data_list:
        list_image_paths = []
        with open(args.data_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if args.data_prefix is not None:
                    each_line = os.path.join(args.data_prefix, line.strip()) 
                else:
                    each_line = line.strip()
                list_image_paths.append(each_line)
    else:
        raise NotImplementedError

    return list_image_paths

def get_model(args):
    from paddle.vision.models import resnet50, resnet101, resnet152, vgg16, mobilenet_v3_large

    from models import ViT_base_patch16_224, ViT_small_patch16_224, ViT_large_patch16_224, MAE_ViT_base_patch16_224

    model_init_args = {'pretrained': True, 'num_classes': args.num_classes}
    if args.model_weights is not None:
        model_init_args['pretrained'] = False

    models_dict = {
        'resnet50': resnet50,  # smoothgrad, intgrad, gradcam
        'resnet101': resnet101,  # smoothgrad, intgrad, gradcam
        'resnet152': resnet152,  # smoothgrad, intgrad, gradcam
        'vgg16': vgg16,  # smoothgrad, intgrad, gradcam
        'mobilenet': mobilenet_v3_large,  # smoothgrad, intgrad, gradcam
        'vit_small': ViT_small_patch16_224,  # smoothgrad, intgrad, ga, bt-token, bt-head
        'vit_base': ViT_base_patch16_224,  # smoothgrad, intgrad, ga, bt-token, bt-head
        'vit_large': ViT_large_patch16_224,  # smoothgrad, intgrad, ga, bt-token, bt-head
        'mae_vit': MAE_ViT_base_patch16_224  # smoothgrad, intgrad, ga-4, ga-6, ga-8, bt-token-4, bt-token-6, bt-token-8, bt-head-4, bt-head-6, bt-head-8
    }
    if 'lrp' == args.it:
        from tutorials.assets.lrp_model import resnet50 as resnet50_lrp
        from tutorials.assets.lrp_model import resnet101 as resnet101_lrp
        from tutorials.assets.lrp_model import resnet152 as resnet152_lrp
        from tutorials.assets.lrp_model import vgg16 as vgg16_lrp
        
        models_dict = {
            'resnet50': resnet50_lrp,
            'resnet101': resnet101_lrp,
            'resnet152': resnet152_lrp,
            'vgg16': vgg16_lrp,
            'mobilenet': None,
            'vit': None,  # to add.
            'mae_vit': None  # to add.            
        }
        assert args.model.lower() in models_dict, "LRP supports resnet and vgg only."

    paddle_model = models_dict[args.model.lower()](**model_init_args)
    
    ## load weights if given
    if args.model_weights is not None:
        state_dict = paddle.load(args.model_weights)
        paddle_model.set_dict(state_dict)
        print("Load weights from", args.model_weights)
    return paddle_model


def get_it(args, paddle_model):
    interpreters_dict = {
        'lime': it.LIMECVInterpreter,
        'gradcam': it.GradCAMInterpreter,
        'intgrad': it.IntGradCVInterpreter,
        'smoothgrad': it.SmoothGradInterpreter,
        'gradshap': it.GradShapCVInterpreter,
        'scorecam': it.ScoreCAMInterpreter,
        'glime': it.GLIMECVInterpreter,
        'lrp': it.LRPCVInterpreter,
        'ga': it.GACVInterpreter,
        'bt': it.BTCVInterpreter
    }

    return interpreters_dict[args.it](paddle_model, device=args.device)


def main(args):
    # get data
    list_image_paths = get_data(args)
    print(args.data_list)
    print(len(list_image_paths))

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

    exp_results = {}
    if os.path.exists(f'./work_dirs/{get_exp_id(args)}.npz'):
        logging.info(f"Loading computed results from ./work_dirs/{get_exp_id(args)}.npz")
        exp_results = dict(np.load(f'./work_dirs/{get_exp_id(args)}.npz', allow_pickle=True))
    
    for i, img_path in enumerate(tqdm(list_image_paths, leave=True, position=0)):
        if img_path in exp_results:
            # load computed exp.
            continue
            exp_result = exp_results[img_path].item()
        else:
            # compute exp. lime_results or array_exp.
            exp = interpreter.interpret(img_path, **it_configs, **img_resize_configs, visual=False)
            if hasattr(interpreter, 'lime_results'):
                exp_result = interpreter.lime_results            
            else:
                exp_result = {'exp': exp}

            exp_results[img_path] = copy.deepcopy(exp_result)
        
        if args.save_eval_result and i % (len(list_image_paths)//50) == 0:
            np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **exp_results)
            logging.info(f'Saving exp_results at {i}')
    
    if args.save_eval_result:
        np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **exp_results)
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
