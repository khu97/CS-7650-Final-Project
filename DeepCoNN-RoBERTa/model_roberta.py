import torch
from torch import nn
from transformers import RobertaModel


class RoBERTaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        # Freeze RoBERTa parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.output = nn.Linear(self.model.config.hidden_size, config.cnn_out_dim)

    def forward(self, token_ids, attention_mask):
        # token_ids, attention_mask: (new_batch_size, seq_len)
        out = self.model(input_ids=token_ids, attention_mask=attention_mask)
        cls_rep = out.last_hidden_state[:, 0, :]         # CLS embedding
        return self.output(cls_rep)                      # (new_batch_size, cnn_out_dim)


class FactorizationMachine(nn.Module):

    def __init__(self, p, k):  # p=cnn_out_dim
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output  # out shape(batch_size, 1)


class DeepCoNN(nn.Module):

    def __init__(self, config):
        super(DeepCoNN, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.encoder_u = RoBERTaEncoder(config)
        self.encoder_i = RoBERTaEncoder(config)
        self.fm = FactorizationMachine(config.cnn_out_dim * 2, 10)

    def forward(self, user_tokens, user_attention, item_tokens, item_attention):  # input shape(batch_size, review_count, review_length)
        # user reviews
        b, r, L = user_tokens.shape
        user_tokens = user_tokens.reshape(b*r, L)
        user_attention = user_attention.reshape(b*r, L)
        user_latent = self.encoder_u(user_tokens, user_attention)  # (b*r, out_dim)
        user_latent = user_latent.reshape(b, r, -1).mean(dim=1)            # aggregate reviews

        # item reviews
        item_tokens = item_tokens.reshape(b*r, L)
        item_attention = item_attention.reshape(b*r, L)
        item_latent = self.encoder_i(item_tokens, item_attention)
        item_latent = item_latent.reshape(b, r, -1).mean(dim=1)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        prediction = torch.sigmoid(prediction) * 4 + 1  # Scale to [1, 5]
        return prediction
