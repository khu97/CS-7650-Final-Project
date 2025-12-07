import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


class TinyBERTDeepCoNNDataset(Dataset):
    def __init__(self, csv_file, config):
        self.config = config

        print(date(), "## Loading CSV:", csv_file)
        df = pd.read_csv(csv_file, header=None)
        df.columns = ['user', 'item', 'review', 'rating']
        df['review'] = df['review'].astype(str)

        # Build user and item review dictionaries
        self.user_reviews = df.groupby('user')['review'].apply(list).to_dict()
        self.item_reviews = df.groupby('item')['review'].apply(list).to_dict()

        # Rows for training
        self.rows = df[['user', 'item', 'rating']].values

        print(date(), "## Loading tokenizer:", config.bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)

        # === PRE-TOKENIZATION STEP ===
        print(date(), "## Pre-tokenizing user texts...")
        self.user_tokens = {}
        for user, reviews in self.user_reviews.items():
            text = " ".join(reviews)
            enc = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=config.max_seq_len,
                return_tensors='pt'
            )
            self.user_tokens[user] = (
                enc['input_ids'][0],
                enc['attention_mask'][0]
            )

        print(date(), "## Pre-tokenizing item texts...")
        self.item_tokens = {}
        for item, reviews in self.item_reviews.items():
            text = " ".join(reviews)
            enc = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=config.max_seq_len,
                return_tensors='pt'
            )
            self.item_tokens[item] = (
                enc['input_ids'][0],
                enc['attention_mask'][0]
            )

        print(date(), "## Pre-tokenization complete!")


    def __len__(self):
        return len(self.rows)


    def __getitem__(self, idx):
        user, item, rating = self.rows[idx]

        # Retrieve pre-tokenized tensors
        u_ids, u_mask = self.user_tokens[user]
        i_ids, i_mask = self.item_tokens[item]

        return (
            u_ids.clone(),     # clone() avoids shared memory issues in DataLoader
            u_mask.clone(),
            i_ids.clone(),
            i_mask.clone(),
            torch.tensor(rating, dtype=torch.float)
        )


def predict_mse(dataloader, model, device):
    model.eval()
    mses = []

    with torch.no_grad():
        for batch in dataloader:
            u_ids, u_mask, i_ids, i_mask, ratings = batch

            u_ids = u_ids.to(device)
            u_mask = u_mask.to(device)
            i_ids = i_ids.to(device)
            i_mask = i_mask.to(device)
            ratings = ratings.to(device)

            preds = model(u_ids, u_mask, i_ids, i_mask)
            mse = torch.mean((preds - ratings) ** 2).item()
            mses.append(mse)

    return float(np.mean(mses))
