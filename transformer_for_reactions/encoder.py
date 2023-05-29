import torch 
import torch.nn as nn

torch.manual_seed(1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embeddings = nn.Embedding(input_dim+2, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        embedded = self.dropout(self.embeddings(src))

        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell