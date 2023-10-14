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
from training_args import args


def get_model(args):
    from paddle.vision.models import resnet50, resnet101, resnet152, vgg16, mobilenet_v3_large
    from models import ViT_base_patch16_224, ViT_small_patch16_224, ViT_large_patch16_224
    from model_mae_vit import vit_base_patch16 as MAE_ViT_base_patch16_224

    model_init_args = {'pretrained': False, 'num_classes': args.num_classes}

    models_dict = {
        'resnet50': resnet50,  # smoothgrad, intgrad, gradcam
        'resnet101': resnet101,  # smoothgrad, intgrad, gradcam
        'resnet152': resnet152,  # smoothgrad, intgrad, gradcam
        'vgg16': vgg16,  # smoothgrad, intgrad, gradcam
        'mobilenet': mobilenet_v3_large,  # smoothgrad, intgrad, gradcam
        'vit_small': ViT_small_patch16_224,  # smoothgrad, intgrad, ga, bt-token, bt-head
        'vit_base': ViT_base_patch16_224,  # smoothgrad, intgrad, ga, bt-token, bt-head
        'vit_large': ViT_large_patch16_224,  # smoothgrad, intgrad, ga, bt-token, bt-head
        'mae_vit': MAE_ViT_base_patch16_224  # smoothgrad, intgrad, ga, bt-token, bt-head
    }
    paddle_model = models_dict[args.model_name.lower()](**model_init_args)
    
    return paddle_model    


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
    train_loader, val_loader = get_dataloaders(args)

    # model
    model = get_model(args)
    logging.info(model)

    # training 
    trainable_parameters = model.parameters()
    ## default lr policy and hard code.
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=[int(2.0*args.epochs/3.0)],
        values=[args.lr, args.lr*0.1]
    )
    optimizer = getattr(pp_optimizer, args.optimizer)(
        parameters=trainable_parameters,
        learning_rate=lr_scheduler
    )
    cross_entropy = nn.CrossEntropyLoss()

    logging.info("Training Started...")
    print(model)

    for epoch in range(args.epochs):
        model.train()
        bar = tqdm(train_loader)
        bar.set_description("Training")
        losses = 0
        for idx, batch in enumerate(bar):
            inputs, targets = batch

            logits = model(inputs)

            loss = cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses += loss.item() * train_loader.batch_size
            acc = (logits.argmax(1) == targets).cast('float32').mean()

            if idx % 100 == 0:
                _log = f" EPOCH| {epoch}, BATCH| {idx}/{len(bar)}, LOSS| {loss.item()}, ACC| {acc}"
                logging.info(_log)
                print(_log)

        _log = f"EPOCH | {epoch} TOTAL LOSS | {losses / len(train_loader.dataset)}"
        logging.info(_log)
        print(_log)

        loss, acc, cm = evaluate(model, val_loader)
        _log = f"EVAL EPOCH | {epoch} ACC | {acc}, LOSS | {loss}\n CM: {cm}"
        logging.info(_log)
        print(_log)
        
        if (epoch + 1) % args.ckpt_num == 0:
            paddle.save(model.state_dict(), f'./results/result_{args.name}/ckpt-{epoch}.pd')

    paddle.save(model.state_dict(), f'./results/result_{args.name}/model.pd')
    
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
        filename=f'./results/result_{args.name}/app_{timestr}.log', 
        filemode='w', 
        format=FORMAT,
        level=getattr(logging, args.log.upper())
    )
    logging.info(f'{args}\n')

    run(args)

