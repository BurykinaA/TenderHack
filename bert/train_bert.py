# Imports
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch.nn as nn
from torch.utils.data import Dataset

from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModel

os.environ["WANDB_DISABLED"] = "true"


class CFG:
    """Config."""
    model_path = 'DeepPavlov/rubert-base-cased-conversational'

    data_path = "train_cleaned.csv"
    path_to_logs = "temp/"
    save_dir = "checkpoints/"
    
    learning_rate = 1e-5
    weight_decay = 0.01
    num_fold = 10
    exec_fold = [0]
    epochs = 5
    batch_size = 8
    
    dropout=0.25

# Getting the data
train_df = pd.read_csv(CFG.data_path)
train_df["len"] = train_df["input"].astype(str).apply(lambda x: len(x.split()))
train_df = train_df[train_df["len"] <= 20]
train_df = train_df.reset_index()

# Building the validation
train_df['kfold'] = 0
kfold = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=0xFACED)
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df, train_df["label"].values)):
    train_df.loc[val_idx, 'kfold'] = fold

# Getting the objects
tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
CFG.tokenizer = tokenizer
CFG.config = AutoConfig.from_pretrained(CFG.model_path)


class TrainDataset(Dataset):
    """Dataset sctructure."""
    def __init__(self, df):
        self.inputs = df['input'].values.astype(str)
        self.label = df['label'].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]
        label = self.label[item]
        
        m = {
        **CFG.tokenizer(inputs),
        'label':label.astype(np.float32)
        }
        
        return m


class LesGoNet(nn.Module):
    """Model structure."""
    def __init__(self, model_name, pretrained=True):
        super(LesGoNet, self).__init__()
        self.config = CFG.config
        self.model_name = model_name
        
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.dropout0 = nn.Dropout(p=CFG.dropout)
        
        self.fc = nn.Linear(self.config.hidden_size, 83)
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=8),
                                                 num_layers=1)


    def forward(self, input_dict):     
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        last_hidden_states = outputs[0]
        feature = self.transformer(last_hidden_states)
        feature = feature[:,0,:].squeeze(1) 
        
        logits = self.fc(self.dropout0(feature))
        
        logits = logits.squeeze(-1)
        out_dict = {
            'logits' : logits,
        }
            
        return out_dict


def compute_metrics(eval_pred):
    """Custom metric."""
    predictions, labels = eval_pred
    return {"f1_score": f1_score(labels, np.argmax(predictions, axis=1), average='macro')}


class CustomTrainer(Trainer):
    """Custom loss."""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(inputs)
        logits = outputs.get('logits')
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels.long())
        return (loss, outputs) if return_outputs else loss


# Train loop
oof_df = pd.DataFrame()
for fold in range(CFG.num_fold):
    if fold in CFG.exec_fold:
        tr_data = train_df[train_df['kfold']!=fold].reset_index(drop=True)
        va_data = train_df[train_df['kfold']==fold].reset_index(drop=True)
        tr_dataset = TrainDataset(tr_data)
        va_dataset = TrainDataset(va_data)

        args = TrainingArguments(
            output_dir=CFG.path_to_logs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=CFG.learning_rate,
            per_device_train_batch_size=CFG.batch_size,
            per_device_eval_batch_size=CFG.batch_size,
            num_train_epochs=CFG.epochs,
            weight_decay=CFG.weight_decay,
            metric_for_best_model="f1_score",
            load_best_model_at_end=True,
            dataloader_num_workers=16,
        )

        model = LesGoNet(CFG.model_path)
        
        trainer = CustomTrainer(
            model,
            args,
            train_dataset=tr_dataset,
            eval_dataset=va_dataset,
            tokenizer=CFG.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        shutil.rmtree(CFG.path_to_logs)
        trainer.save_model(os.path.join(CFG.save_dir, f"model_{fold}"))

        outputs = trainer.predict(va_dataset)
        predictions = outputs.predictions.reshape(-1)
        va_data['preds'] = predictions
        oof_df = pd.concat([oof_df, va_data])


# Calculating OOF
predictions = oof_df['preds'].values
label = oof_df['overall_worklogs'].values
eval_pred = predictions, label
compute_metrics(eval_pred)
oof_df.to_csv('oof_df.csv')