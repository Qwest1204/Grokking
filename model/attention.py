import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, "d_model must be divisible by num_head"

        self.d_k = d_model // num_head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(2, 3)) / math.sqrt(d_k)
        attention_score = attention_score.softmax(dim=-1)
        attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v):
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = q.view(q.shape[0], q.shape[1], self.num_head, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_head, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_head, self.d_k).transpose(1, 2)

        x, attention_score = MHA.attention(q, k, v, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_head*self.d_k)

        return self.W_o(x), attention_score
