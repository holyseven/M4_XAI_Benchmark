


import os
import time
import logging
import numpy as np
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.optimizer.lr import LinearWarmup

import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

# local imports
from data import SST2, MovieReview
from training_args import args


def get_exp_id(args):
    return f'{args.model_name}_{args.name}'

def get_model(args):
    model_name = args.model_name

    if model_name == 'lstm':
        return
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def get_dataloaders(args, tokenizer):

    def text_to_input(raw_text):
        encoded_inputs = tokenizer(text=raw_text, max_seq_len=args.max_seq_length)
        _batched_and_to_tuple = tuple([np.array(v) for v in encoded_inputs.values()])
        return _batched_and_to_tuple

    if 'SST-2' in args.dataset_root:
        train_set = SST2(args.dataset_root, split_name='train', text_to_input_fn=text_to_input)
        dev_set = SST2(args.dataset_root, split_name='dev', text_to_input_fn=text_to_input)
    elif 'movies' in args.dataset_root:
        train_set = MovieReview(args.dataset_root, split_name='train', text_to_input_fn=text_to_input)
        dev_set = MovieReview(args.dataset_root, split_name='val', text_to_input_fn=text_to_input)
    else:
        raise ValueError(f"unknown dataset: {args.dataset_root}")
    
    def batchify_fn(samples):
        stack = Stack(dtype="int64")
        labels = stack([label for encoded_input, label in samples])

        pad1 = Pad(axis=0, pad_val=tokenizer.pad_token_id)
        r1 = pad1([encoded_input[0] for encoded_input, label in samples])

        if len(samples[0][0]) > 1:
            pad2 = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
            r2 = pad2([encoded_input[1] for encoded_input, label in samples])
            return (r1, r2), labels
        else:
            return (r1, ), labels

    # batchify_fn = lambda samples, fn=Tuple(
    #     Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    #     Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    #     Stack(dtype="int64")  # label
    # ): [data for data in fn(samples)]

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=16, collate_fn=batchify_fn)
    dev_loader = DataLoader(dev_set, shuffle=False, batch_size=args.batch_size_eval, num_workers=8, collate_fn=batchify_fn)

    return train_loader, dev_loader

def evaluate(model, loader):
    fin_targets = []
    fin_probabs = []
    
    with paddle.no_grad():
        model.eval()
        epoch_loss = 0
        for batch in tqdm(loader, unit="batches", desc="Evaluating"):
            inputs, targets = batch

            logits = model(*inputs)

            loss = F.cross_entropy(logits, targets)
            epoch_loss += loss.item() * loader.batch_size

            fin_targets.extend(targets.tolist())
            fin_probabs.extend(F.softmax(logits, axis=1).numpy())

        loss = epoch_loss / len(loader.dataset)
        fin_preds = np.argmax(np.array(fin_probabs), axis=-1)

    acc = (fin_preds == fin_targets).mean()

    return loss, acc


def run(args):
    # model
    model, tokenizer = get_model(args)
    logging.info(model)

    # data
    train_loader, val_loader = get_dataloaders(args, tokenizer)

    learning_rate = args.lr # default 5e-5
    epochs = args.epochs
    warmup_proportion = 0.1
    weight_decay = 0.01

    # training
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
            ]
        )

    cross_entropy = nn.CrossEntropyLoss()

    logging.info("Training Started...")
    print(model)

    for epoch in range(epochs):
        model.train()
        bar = tqdm(train_loader)
        bar.set_description("Training")
        losses = 0
        for idx, batch in enumerate(bar):
            inputs, targets = batch

            logits = model(*inputs)

            loss = cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
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

        loss, acc = evaluate(model, val_loader)
        _log = f"EVAL EPOCH | {epoch} ACC | {acc}, LOSS | {loss}\n"
        logging.info(_log)
        print(_log)

    paddle.save(model.state_dict(), f'./results/{get_exp_id(args)}/model.pd')
    
    return


if __name__ == '__main__':    
    device = args.device
    paddle.set_device(device)
    np.random.seed(args.random_seed)
    
    os.makedirs('./results', exist_ok=True)
    os.makedirs(f'./results/{get_exp_id(args)}', exist_ok=True)
    FORMAT = '%(asctime)-15s %(message)s'
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        filename=f'./results/{get_exp_id(args)}/app_{timestr}.log', 
        filemode='w', 
        format=FORMAT,
        level=getattr(logging, args.log.upper())
    )
    logging.info(f'{args}\n')

    run(args)

