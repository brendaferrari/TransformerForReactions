import torch 
import torch.nn as nn
import random

torch.manual_seed(1)

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):

        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]

        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)

        enc_src = self.encoder(src)

        input = tgt[0,:]


        for t in range(1, tgt_len):
            output = self.decoder(input, enc_src)
            output = output[t]


            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio

            #top1 = output.argmax(1)
            top1 = torch.argmax(output, dim=1)

            input = tgt[t] if teacher_force else top1

        return outputs