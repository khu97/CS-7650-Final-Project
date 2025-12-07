import os
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from model_roberta import DeepCoNN
from utils import load_embedding, DeepCoNNDataset, predict_mse, date, tokenize_reviews, custom_collate_fn, calculate_rating_weights
from transformers import AutoTokenizer
from tqdm import tqdm


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    rating_weights = calculate_rating_weights(config.train_file)
    model.eval()
    train_mse = predict_mse(model, train_dataloader, config.device)
    valid_mse = predict_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()  
        total_loss, total_samples = 0, 0
        # batch_losses = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.train_epochs}")
        for batch in pbar:
            user_reviews, item_reviews, ratings = batch
            
            # predict = model(user_reviews, item_reviews)

            user_tokens, user_attention, item_tokens, item_attention = tokenize_reviews(
                user_reviews, item_reviews,
                tokenizer,
                max_len=config.max_seq_len)
            predict = model(user_tokens.to(config.device), user_attention.to(config.device), 
                            item_tokens.to(config.device), item_attention.to(config.device))

            loss = F.mse_loss(predict, ratings.to(config.device), reduction='sum')  
            opt.zero_grad()  
            loss.backward()  
            opt.step()  

            # batch_loss = loss.item() / len(predict)
            # batch_losses.append(batch_loss)
            # # Print when we hit a new best batch loss
            # if batch_loss == min(batch_losses):
            #     print(f"  New best batch loss: {batch_loss:.6f}")

            total_loss += loss.item()
            total_samples += len(predict)

            # Update progress bar with current training loss
            current_mse = total_loss / total_samples
            pbar.set_postfix({'train_mse': f'{current_mse:.6f}'})

        lr_sch.step()
        model.eval()  
        valid_mse = predict_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model, config):
    print(f'{date()}## Start the testing!')
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    start_time = time.perf_counter()
    test_loss = predict_mse(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(config)
    print(f'{date()}## Load embedding and data...')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    train_dataset = DeepCoNNDataset(config.train_file, word_dict, config)
    valid_dataset = DeepCoNNDataset(config.valid_file, word_dict, config, retain_rui=True)
    test_dataset = DeepCoNNDataset(config.test_file, word_dict, config, retain_rui=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=custom_collate_fn)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=custom_collate_fn)

    model = DeepCoNN(config).to(config.device)
    del train_dataset, valid_dataset, test_dataset, word_emb, word_dict

    os.makedirs(os.path.dirname(config.model_file), exist_ok=True) 
    train(train_dlr, valid_dlr, model, config, config.model_file)
    # test(test_dlr, torch.load(config.model_file), config)
