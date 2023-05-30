import torch 
import torch.nn as nn
from transformer_for_reactions.attention import MultiHeadAttention
from transformer_for_reactions.feed_forward import FeedFoward

torch.manual_seed(1)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, emb_dim, n_head, dropout):
        super().__init__()

        head_size = emb_dim // n_head
        self.sa = MultiHeadAttention(head_size, n_head, emb_dim, dropout)
        self.enc_dec_a = MultiHeadAttention(head_size, n_head, emb_dim, dropout)
        self.ffwd = FeedFoward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln3 = nn.LayerNorm(emb_dim)

    def forward(self, dec, enc):

        if enc is not None:
            dec = dec + self.sa(self.ln1(dec), None)
            dec = dec + self.enc_dec_a(self.ln2(enc),self.ln1(dec))
            dec = dec + self.ffwd(self.ln3(dec))
        else:
            dec = dec + self.sa(self.ln1(dec))
            dec = dec + self.ffwd(self.ln2(dec))

        return dec

class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.transformer = nn.ModuleList([TransformerDecoderLayer(emb_dim, 4, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, enc_src):
        if input.dim() == 1:
            input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        for layer in self.transformer:
            output = layer(embedded, enc_src)

        prediction = self.fc_out(output)

        return prediction