import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, head_size, emb_dim, dropout):
        super().__init__()

        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(500, 500)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, q):

        B,T,C = x.shape
        k = self.key(x)
        if q is not None:
            q = self.query(q)
        else:
            q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, head_size, num_heads, emb_dim, dropout):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size, emb_dim, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, q):

        out = torch.cat([h(x, q) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out