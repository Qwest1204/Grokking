import torch
import torch.nn as nn

from attention import MHA
from modules import MLP, ProjectionLayer, ResidualConnection, InputEmbedding

class GrokkingTransformer(nn.Module):
    """
    Simple Grokking Transformer with one layer of MHA & MLP
    """
    def __init__(self, d_model: int, vocab_size: int, d_ff: int, dropout: float, num_heads: int) -> None:
        super().__init__()
        self.embedding = InputEmbedding(d_model, vocab_size=vocab_size)
        self.mha = MHA(d_model, num_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.residual_connection = ResidualConnection(dropout)
        self.projection = ProjectionLayer(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, att = self.mha(x, x, x)
        x = self.residual_connection(x)
        x = self.mlp(x)
        x = self.residual_connection(x)
        x_proj = self.projection(x)
        logits = x_proj[:, -1, :]
        x = torch.log_softmax(logits, dim=-1)
        return x, att
