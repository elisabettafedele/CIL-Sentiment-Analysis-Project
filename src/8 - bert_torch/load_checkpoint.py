from sympy import arg
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Subset
from torchvision.datasets import DatasetFolder
import numpy as np
import os

checkpoint = "path/to/checkpoint"
test_path = "path/to/test/dir"
pred_path = "path/to/pred.csv"

def loaddata(fpath):
    with open(fpath, "r") as f:
        data = f.readlines()
        return data[0]

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

test_dataset = get_dataloader_test(test_path, tokenizer)

print(len(test_dataset))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)

training_args = TrainingArguments(
    per_device_eval_batch_size=32,
    output_dir="predict"
)
trainer = Trainer(
    model = model,
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=training_args
)

results = trainer.predict(test_dataset).predictions

y_preds = np.argmax(results, axis=1)

y_preds = [-1 if val == 0 else 1 for val in y_preds]

import pandas as pd
df = pd.DataFrame(y_preds, columns=["Prediction"])
df.index.name = "Id"
df.index += 1
df.to_csv(pred_path)

with open(os.path.join(checkpoint, "prediction.npy"), "wb") as f:
    np.save(f, results)