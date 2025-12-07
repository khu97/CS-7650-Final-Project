# utils.py

import time
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import Config
from tqdm import tqdm

config = Config()
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def calculate_rating_weights(train_file):
    df = pd.read_csv(train_file, header=None, names=['userID', 'itemID', 'review', 'rating'])
    rating_counts = df['rating'].value_counts().to_dict()
    total = len(df)
    # Inverse frequency weighting
    weights = {int(rating): total / (len(rating_counts) * count) 
               for rating, count in rating_counts.items()}
    print(f"{date()}## Rating distribution in training set:")
    for rating in sorted(rating_counts.keys()):
        print(f"  Rating {int(rating)}: {rating_counts[rating]:5d} samples, weight: {weights[int(rating)]:.4f}")
    return weights

def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())

def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


def tokenize_reviews(user_reviews, item_reviews, tokenizer, max_len=512):
    batch_size = len(user_reviews)
    review_count = len(user_reviews[0])
    
    # Flatten all reviews
    user_reviews_flat = [str(review) for reviews in user_reviews for review in reviews]
    item_reviews_flat = [str(review) for reviews in item_reviews for review in reviews]

    # print(type(user_reviews), len(user_reviews))
    # print(user_reviews[0])
    # print(len(user_reviews[0]))

    # print(type(user_reviews_flat[0]))
    # print(len(user_reviews_flat))
    # print(user_reviews_flat[0])
    
    # Tokenize all at once
    user_encoded = tokenizer(user_reviews_flat, padding='max_length', 
                            truncation=True, max_length=max_len, 
                            return_tensors='pt')
    item_encoded = tokenizer(item_reviews_flat, padding='max_length',
                            truncation=True, max_length=max_len,
                            return_tensors='pt')
    
    # Reshape to (batch_size, review_count, max_len)
    user_tokens = user_encoded['input_ids'].view(batch_size, review_count, -1)
    user_attention = user_encoded['attention_mask'].view(batch_size, review_count, -1)
    item_tokens = item_encoded['input_ids'].view(batch_size, review_count, -1)
    item_attention = item_encoded['attention_mask'].view(batch_size, review_count, -1)
    
    return user_tokens, user_attention, item_tokens, item_attention


def predict_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    pbar = tqdm(dataloader, desc="Evaluating")
    # print(device)
    with torch.no_grad():
        for batch in pbar:
            user_reviews, item_reviews, ratings = batch
            ratings = ratings.to(device)
            user_tokens, user_attention, item_tokens, item_attention = tokenize_reviews(
                    user_reviews, item_reviews, tokenizer, max_len=config.max_seq_len)
            predict = model(user_tokens.to(device), user_attention.to(device),
                            item_tokens.to(device), item_attention.to(device))
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
            pbar.set_postfix({'MSE': f'{mse/sample_count:.4f}'})
    return mse / sample_count


def custom_collate_fn(batch):
    user_reviews = [item[0] for item in batch]  # list of lists of strings
    item_reviews = [item[1] for item in batch]  # list of lists of strings
    ratings = torch.stack([item[2] for item in batch])  # stack the rating tensors
    
    return user_reviews, item_reviews, ratings


class DeepCoNNDataset(Dataset):
    def __init__(self, data_path, word_dict, config, retain_rui=True):
        self.word_dict = word_dict
        self.config = config
        self.retain_rui = retain_rui  
        self.PAD_WORD_idx = self.word_dict[config.PAD_WORD]
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        # df['review'] = df['review'].apply(self._review2id)  
        self.sparse_idx = set()  
        user_reviews = self._get_reviews(df)  
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_reviews = [user_reviews[idx] for idx in range(len(user_reviews)) if idx not in self.sparse_idx]
        self.item_reviews = [item_reviews[idx] for idx in range(len(item_reviews)) if idx not in self.sparse_idx]
        self.rating = rating[[idx for idx in range(len(rating)) if idx not in self.sparse_idx]]


    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]
            if self.retain_rui:
                reviews = df_data['review'].to_list()  # list of strings
            else:
                reviews = df_data['review'][df_data[costar] != costar_id].to_list()
            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        return lead_reviews

    def _adjust_review_list(self, reviews, r_length, r_count):
        # Truncate number of reviews to r_count
        reviews = reviews[:r_count]  
        # Pad missing reviews with empty string
        reviews += [""] * (r_count - len(reviews))  
        return reviews

    # def _review2id(self, review): 
    #     if not isinstance(review, str):
    #         return []  
    #     wids = []
    #     for word in review.split():
    #         if word in self.word_dict:
    #             wids.append(self.word_dict[word])  
    #         else:
    #             wids.append(self.PAD_WORD_idx)
    #     return wids
