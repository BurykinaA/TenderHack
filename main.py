from typing import Union
import pandas as pd
import numpy as np
import json
import pymorphy2
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModel


class LesGoNet(nn.Module):
    """Model structure."""

    def __init__(self, model_name, pretrained=True):
        super(LesGoNet, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_name = model_name

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.dropout0 = nn.Dropout(p=0.25)
        self.loss = False

        self.fc = nn.Linear(self.config.hidden_size, 83)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=8),
            num_layers=1)

    def forward(self, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        last_hidden_states = outputs[0]
        feature = self.transformer(last_hidden_states)
        feature = feature[:, 0, :].squeeze(1)

        logits = self.fc(self.dropout0(feature))

        logits = logits.squeeze(-1)
        out_dict = {
            'logits': logits,
        }

        return out_dict


class CustomTrainer(Trainer):
    """Custom loss."""

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss_fn = nn.MSELoss()
        loss = loss_fn(torch.FloatTensor([1]), torch.FloatTensor([1]))
        return (loss, outputs) if return_outputs else loss


class GlobalObjects:
    lemmatizer = pymorphy2.MorphAnalyzer()
    stop_words = set(stopwords.words('russian'))
    special_symbols = string.ascii_uppercase + "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890-"
    with open('codes.json', 'r') as file:
        decoding = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained('model')
    model = LesGoNet('bert-base-multilingual-cased', pretrained=False)
    model.load_state_dict(torch.load("model/pytorch_model.bin", map_location=torch.device('cpu')))
    trainer = CustomTrainer(model, tokenizer=tokenizer)
    del model

    min_digit_len = 4
    is_serial_thr = 0.5

    clustering_threshold = 0.9
    max_samples = 10


class InferDataset(Dataset):
    def __init__(self, df):
        self.inputs = df['input'].values.astype(str)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item, tokenizer=GlobalObjects.tokenizer):
        inputs = self.inputs[item]

        return {
            **tokenizer(inputs, inputs, truncation=True),
            'label': -1
        }


def process_input(inp: str, lemmatizer=GlobalObjects.lemmatizer) -> str:
    """Process the input."""
    # Cleaning
    inp = inp.lower()
    inp = [i.strip(string.punctuation) for i in inp.split(' ')]
    inp = [i for i in inp if i not in GlobalObjects.stop_words]
    inp = " ".join(inp)

    # Lemmatization
    inp = inp.split()
    inp = ' '.join([lemmatizer.parse(word)[0].normal_form
                    for word in inp])

    return inp


def look_for_cluster(inp: str, df: pd.DataFrame,
                     trainer=GlobalObjects.trainer,
                     decoding=GlobalObjects.decoding) -> pd.DataFrame:
    """Look for possible cluster."""
    te_dataset = InferDataset(pd.DataFrame({"input": [inp]}))
    outputs = trainer.predict(te_dataset)
    pred = np.array(outputs.predictions).argmax(axis=1).item()
    # df["is_in_cluster"] = df["label"] == pred
    return df[df["label"] == pred].index.to_list(), decoding[str(pred)]


def isSerial(input_string: str,
             threshold=GlobalObjects.is_serial_thr,
             special_symbols=GlobalObjects.special_symbols) -> bool:
    counter = 0
    for x in input_string:
        if x in special_symbols:
            counter += 1

    return counter / len(input_string) > threshold


def look_for_model_name(inp: str,
                        min_digit_len=GlobalObjects.min_digit_len) -> Union[str, None]:
    """Look for model name."""

    words = inp.split()
    output = ""
    for word, is_serial in dict(zip(words, list(map(isSerial, words)))).items():
        word = word.replace(",", " ")
        if is_serial:
            if word.isdigit():
                if len(word) >= min_digit_len:
                    output += f"{word} "
            else:
                output += f"{word} "

    if output.isdigit() and len(output) < min_digit_len:
        return None
    if len(output) < 3:
        return None

    return output[:-1] if output != "" else None


def apply_tf_idf(inp: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply TF-IDF."""
    return df


def search_by_model_name(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Search for the samples with the same model name."""
    df = df[np.array([model_name in i for i in df["input"].astype(str).values])]
    return df

df1 = pd.read_csv(r'C:\Users\alina\Downloads\train_cleaned.csv')

def run_search_engine(inp: str, df=df1, max_samples=GlobalObjects.max_samples):
    """The search engine pipeline."""

    # Processing input.
    inp = process_input(inp)

    # Looking for cluster.
    df, category = look_for_cluster(inp, df)
    # print(df)


    # Looking for model name.
    model_name = look_for_model_name(inp)

    # Checking if model name exists.
    # if model_name is None:
    #     # Applying TF-IDF.
    #     return [], category
    #else:
        # Searching for the samples with the same model name.
        #df = search_by_model_name(model_name, df)
    if len(df) == 0:
        return [], category
    else:
        return df, category



