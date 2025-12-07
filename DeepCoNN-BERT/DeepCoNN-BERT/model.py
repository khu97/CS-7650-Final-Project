import torch
from torch import nn
from transformers import AutoModel



class FactorizationMachine(nn.Module):

    def __init__(self, p, k):
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)  
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output  


class TinyBERTDeepCoNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = AutoModel.from_pretrained(config.bert_model_name)

        if config.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        self.user_proj = nn.Linear(hidden_size, config.cnn_out_dim)
        self.item_proj = nn.Linear(hidden_size, config.cnn_out_dim)

        self.fm = FactorizationMachine(config.cnn_out_dim * 2, 10)

    def forward(self, u_ids, u_mask, i_ids, i_mask):
        u_output = self.bert(input_ids=u_ids, attention_mask=u_mask)
        u_hidden = u_output.last_hidden_state
        u_vec = torch.mean(u_hidden, dim=1)

        i_output = self.bert(input_ids=i_ids, attention_mask=i_mask)
        i_hidden = i_output.last_hidden_state
        i_vec = torch.mean(i_hidden, dim=1)

        u_latent = self.user_proj(u_vec)
        i_latent = self.item_proj(i_vec)

        concat = torch.cat([u_latent, i_latent], dim=1)

        pred = self.fm(concat)
        return pred.squeeze(1)

