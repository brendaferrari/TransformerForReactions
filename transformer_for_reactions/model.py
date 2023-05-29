import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from transformer_for_reactions.encoder import Encoder
from transformer_for_reactions.decoder import Decoder
from transformer_for_reactions.seq2seq import Seq2Seq


class Model:

    def __init__(self, INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, 
                    N_LAYERS, ENC_DROPOUT, DEC_DROPOUT):

        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM
        self.ENC_EMB_DIM = 256
        self.DEC_EMB_DIM = 256
        self.HID_DIM = 512
        self.N_LAYERS = 2
        self.ENC_DROPOUT = 0.5
        self.DEC_DROPOUT = 0.5

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
            
    def count_parameters(self, model_seq2seq):
        return sum(p.numel() for p in model_seq2seq.parameters() if p.requires_grad)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs  

    def train(self, model, x_iterator, y_iterator, optimizer, criterion, clip):

        model.train()
        epoch_loss = 0

        for i, batch in enumerate(x_iterator):
            src = x_iterator[i]
            tgt = y_iterator[i]

            optimizer.zero_grad()

            output = model(src, tgt)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)

            loss = criterion(output, tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
        
        return epoch_loss/len(x_iterator)

    def evaluate(self, model, x_iterator, y_iterator, criterion):
        
        model.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
        
            for i, batch in enumerate(x_iterator):

                src = x_iterator[i]
                tgt = y_iterator[i]


                output = model(src, tgt, 0)

                output_dim = output.shape[-1]
                
                output = output[1:].view(-1, output_dim)
                tgt = tgt[1:].view(-1)

                loss = criterion(output, tgt)
                
                epoch_loss += loss.item()
            
        return epoch_loss / len(x_iterator)


    def model_run(self, x_train_iterator, y_train_iterator, 
                  x_valid_iterator, y_valid_iterator, N_EPOCHS):

        enc = Encoder(self.INPUT_DIM, self.ENC_EMB_DIM, self.HID_DIM, 
                        self.N_LAYERS, self.ENC_DROPOUT)
        dec = Decoder(self.OUTPUT_DIM, self.DEC_EMB_DIM, self.HID_DIM, 
                        self.N_LAYERS, self.DEC_DROPOUT)

        model_seq2seq = Seq2Seq(enc, dec, self.device).to(self.device)

        model_seq2seq.apply(self.init_weights)
        print(model_seq2seq)

        print(f'The model has {self.count_parameters(model_seq2seq):,} trainable parameters')

        optimizer = optim.Adam(model_seq2seq.parameters())

        #TRG_PAD_IDX = {'<sos>':1, '<eos>':2, '<pad>':0}
        criterion = nn.CrossEntropyLoss() #ignore_index = TRG_PAD_IDX add latter

        N_EPOCHS = N_EPOCHS
        CLIP = 1

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            
            start_time = time.time()
            
            train_loss = self.train(model_seq2seq, x_train_iterator, y_train_iterator, optimizer, criterion, CLIP)
            valid_loss = self.evaluate(model_seq2seq, x_valid_iterator, y_valid_iterator, criterion)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model_seq2seq.state_dict(), 'tut1-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    def test_run(self, x_test_iterator, y_test_iterator):

        enc = Encoder(self.INPUT_DIM, self.ENC_EMB_DIM, self.HID_DIM, 
                        self.N_LAYERS, self.ENC_DROPOUT)
        dec = Decoder(self.OUTPUT_DIM, self.DEC_EMB_DIM, self.HID_DIM, 
                        self.N_LAYERS, self.DEC_DROPOUT)

        model_seq2seq = Seq2Seq(enc, dec, self.device).to(self.device)

        criterion = nn.CrossEntropyLoss() #ignore_index = TRG_PAD_IDX add latter

        model_seq2seq.load_state_dict(torch.load('tut1-model.pt'))

        test_loss = self.evaluate(model_seq2seq, x_test_iterator, y_test_iterator, criterion)

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
