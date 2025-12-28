import torch
import torch.nn as nn

from model.attention import MHA
from model.modules import MLP, ProjectionLayer, LayerNormalization, InputEmbedding, PositionalEncoding

class GrokkingTransformer(nn.Module):
    """
    Simple Grokking Transformer with one layer of MHA & MLP
    """
    def __init__(self, d_model: int, vocab_size: int, d_ff: int, dropout: float, num_heads: int) -> None:
        super().__init__()
        self.embedding = InputEmbedding(d_model, vocab_size=vocab_size)
        self.mha = MHA(d_model, num_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.pe = PositionalEncoding(d_model, max_len=4)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
        self.projection = ProjectionLayer(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        x, att = self.mha(x, x, x)
        x = x + self.dropout(x)
        x = self.norm1(x)
        x = self.mlp(x)
        x = x + self.dropout(x)
        x = self.norm2(x)
        x_proj = self.projection(x)
        logits = x_proj[:, -1, :]
        return logits, att
