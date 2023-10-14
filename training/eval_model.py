import os
import numpy as np
import logging
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as pp_optimizer
from paddle.io import DataLoader
from paddle.vision import transforms

# local imports
import models
from data import DatasetFolder, Yellow, Yellow1001
from train_model import get_model
from training_args import args


def get_dataloaders(args):
    resize_shorter_edge = args.resize_shorter_edge
    crop_size = args.crop_size

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose(
        [
            transforms.Resize(size=resize_shorter_edge),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(size=crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    # customized
    # transform_train = transforms.Compose(
    #     [
    #         transforms.Resize(size=resize_shorter_edge),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(crop_size),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean, std)
    #     ]
    # )
    # transform_val = transforms.Compose(
    #     [
    #         transforms.Resize(size=(crop_size, crop_size)),
    #         transforms.ToTensor()
    #     ]
    # )

    # train_set = DatasetFolder(os.path.join(args.dataset_root, 'train'), transform=transform_train)
    # val_set = DatasetFolder(os.path.join(args.dataset_root, 'val'), transform=transform_val)

    # logging.info(f"trainset classes: {train_set.classes}")
    # logging.info(f"trainset class2idx: {train_set.class_to_idx}")

    # logging.info(f"val_set classes: {val_set.classes}")
    # logging.info(f"val_set class2idx: {val_set.class_to_idx}")

    train_set = Yellow(args.dataset_root, args.data_list, mode='random', transform=transform_train)
    val_set = Yellow(args.val_dataset_root, args.val_data_list, mode='val', transform=transform_val)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=16)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=args.batch_size_eval, num_workers=8)

    return train_loader, val_loader


def evaluate(model, loader):
    fin_targets = []
    fin_probabs = []
    
    with paddle.no_grad():
        model.eval()
        epoch_loss = 0
        for batch in tqdm(loader, unit="batches", desc="Evaluating"):
            inputs, targets = batch

            logits = model(inputs)

            loss = F.cross_entropy(logits, targets)
            epoch_loss += loss.item() * loader.batch_size

            fin_targets.extend(targets.tolist())
            fin_probabs.extend(F.softmax(logits, axis=1).numpy())

        loss = epoch_loss / len(loader.dataset)
        fin_preds = np.argmax(np.array(fin_probabs), axis=-1)

    acc = (fin_preds == fin_targets).mean()
    acc = accuracy_score(fin_targets, fin_preds)
    cm = confusion_matrix(fin_targets, fin_preds)
    return loss, acc, cm


def run(args):
    # data
    _, val_loader = get_dataloaders(args)

    # model
    model = get_model(args)
    logging.info(model)
    if args.model_weights is not None:
        state_dict = paddle.load(args.model_weights)
        model.set_dict(state_dict)
        print("Load weights from", args.model_weights)

    loss, acc, cm = evaluate(model, val_loader)
    _log = f"EVAL ACC | {acc}, LOSS | {loss}\n CM: {cm}"
    logging.info(_log)
    print(_log)

    return


if __name__ == '__main__':    
    device = args.device
    paddle.set_device(device)
    np.random.seed(args.random_seed)
    
    os.makedirs('./results', exist_ok=True)
    os.makedirs(f'./results/result_{args.name}', exist_ok=True)
    FORMAT = '%(asctime)-15s %(message)s'
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        filename=f'./results/result_{args.name}/app_{timestr}_eval.log', 
        filemode='w', 
        format=FORMAT,
        level=getattr(logging, args.log.upper())
    )
    logging.info(f'{args}\n')

    run(args)

