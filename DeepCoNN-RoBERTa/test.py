import os
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from model_roberta import DeepCoNN
from utils import load_embedding, DeepCoNNDataset, predict_mse, date, tokenize_reviews, custom_collate_fn, tokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

def test(dataloader, model, config):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss = predict_mse(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")

    print(f"\n{date()}## Sample Predictions with Reviews:")
    print("=" * 100)    
    sample_count = 0
    num_examples = 20
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_examples:
                break
            user_reviews, item_reviews, ratings = batch
            user_tokens, user_attention, item_tokens, item_attention = tokenize_reviews(
                user_reviews, item_reviews, tokenizer, max_len=config.max_seq_len)
            predictions = model(user_tokens.to(config.device), user_attention.to(config.device),
                              item_tokens.to(config.device), item_attention.to(config.device))
            for i in range(len(ratings)):
                if predictions[i].item() < 4:
                    if sample_count >= num_examples:
                        break
                    print(f"\nSample {sample_count + 1}:")
                    print(f"  User Review 1: {user_reviews[i][0][:100]}...")  # First 100 chars
                    print(f"  Item Review 1: {item_reviews[i][1][:100]}...")
                    print(f"  Ground Truth:  {ratings[i].item():.2f}")
                    print(f"  Prediction:    {predictions[i].item():.2f}")
                    print(f"  Error:         {abs(ratings[i].item() - predictions[i].item()):.2f}")
                    print("-" * 100)
                    sample_count += 1
    print("=" * 100)

if __name__ == '__main__':
    config = Config()
    print(config)
    print(f'{date()}## Load embedding and data...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    test_dataset = DeepCoNNDataset(config.test_file, word_dict, config, retain_rui=True)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=custom_collate_fn)

    model = DeepCoNN(config).to(config.device)
    del test_dataset, word_emb, word_dict

    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)
    model = torch.load(config.model_file, weights_only=False)
    model.eval()     
    model.to(config.device) 
    test(test_dlr, model, config)
