import torch 
import torch.nn as nn
from transformer_for_reactions.attention import MultiHeadAttention
from transformer_for_reactions.feed_forward import FeedFoward

torch.manual_seed(1)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, emb_dim, n_head, dropout):
        super().__init__()

        head_size = emb_dim // n_head
        self.sa = MultiHeadAttention(head_size, n_head, emb_dim, dropout)
        self.ffwd = FeedFoward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):

        x = x + self.sa(self.ln1(x), None)
        x = x + self.ffwd(self.ln2(x))

        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embeddings = nn.Embedding(input_dim+2, emb_dim)
        self.transformer = nn.ModuleList([TransformerEncoderLayer(emb_dim, 4, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        embedded = self.dropout(self.embeddings(src))

        for layer in self.transformer:
            output = layer(embedded)

        return output