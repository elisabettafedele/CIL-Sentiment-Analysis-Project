import logging
from loguru import logger

import torch
from torch.utils.data import Subset
from torchvision.datasets import DatasetFolder

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

import argparse
import sys
import os

import numpy as np

sys.path.append('')

def loaddata(fpath):
    with open(fpath, "r") as f:
        data = f.readlines()
        return data[0]

def get_dataloaders_train(train_dir, tokenizer):

    def preprocess_function(examples):
        out = tokenizer(examples, truncation=True)
        return out

    def add_label(data):
        values = data[0]
        label = data[1]
        values["label"] = label
        return values

    ds_token = DatasetFolder(train_dir, loaddata, ('.txt'), transform=preprocess_function)

    ds_token_label = list(map(add_label, ds_token))

    train_idx, val_idx = train_test_split(list(range(len(ds_token_label))), test_size=0.2)
    ds = {}
    ds['train'] = Subset(ds_token_label, train_idx)
    ds['val'] = Subset(ds_token_label, val_idx)

    class_names = ds_token.find_classes(train_dir)

    logger.info('Processed Examples:')
    logger.info(ds['train'][0])

    return ds, class_names

def get_dataloader_test(train_dir, tokenizer):
    def preprocess_function(examples):
        out = tokenizer(examples, truncation=True)
        return out

    def add_label(data):
        values = data[0]
        label = data[1]
        values["label"] = label
        return values

    ds_token = DatasetFolder(train_dir, loaddata, ('.txt'), transform=preprocess_function)

    ds_token_label = list(map(add_label, ds_token))

    return ds_token_label



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_name', type=str, help='bert model name', required=True)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('-s', '--seed', type=int, help='random seed', default=42) #TODO add
    parser.add_argument('-d', '--dir', type=str, help='dataset directory', required=True)
    parser.add_argument('-o', '--out', type=str, help='output directory', required=True)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=3)
    parser.add_argument('-l', '--lr', type=float, help='learning rate', default=2e-5)
    parser.add_argument('-w', '--wd', type=float, help='weight decay', default=0.01)
    
    args = parser.parse_args()

    logger.add(
        os.path.join(args.out, 'train.log'),
        level='INFO',
        colorize=False,
    )

    logger.info(f'Input arguments: \n {args}')

    bert_model_name = args.model_name
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')

    train_dir =  os.path.join(args.dir, 'train-seg-full')

    eval_dir = os.path.join(args.dir, "test-seg")

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    twitter_dataset, class_names = get_dataloaders_train(train_dir, tokenizer)

    test_dataset = get_dataloader_test(eval_dir, tokenizer)

    print(len(twitter_dataset["train"]), len(twitter_dataset["val"]))

    print(len(test_dataset))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=len(class_names)).to(device)

    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    logging_steps = len(twitter_dataset["train"]) // args.batch_size
    print(logging_steps)
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.wd,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        logging_strategy="steps",
        save_strategy="epoch",
        warmup_ratio=0.1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=twitter_dataset["train"],
        eval_dataset=twitter_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


    logger.info('Started training')
    trainer.train()
    logger.info('Ended training')

    results = trainer.predict(test_dataset)

    y_preds = np.argmax(results.predictions, axis=1)

    y_preds = [-1 if val == 0 else 1 for val in y_preds]
    
    import pandas as pd
    df = pd.DataFrame(y_preds, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv("test_data.csv")

    exit(0)