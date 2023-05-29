import os
import glob
import itertools
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class Preprocess:

    def get_data(self, file):
        
        d = []
        d_tgt = []
        for f in file:
            with open(f, 'r') as inp:
                rxn = inp.readlines()
                if f.split('_')[1] == 'src.txt':
                    for r in rxn:
                        r = r.strip().split(" ")
                        d_sr={}
                        d_sr['src'] = r
                        d.append(d_sr)
                        
                elif f.split('_')[1] == 'tgt.txt':
                    rxn_len = len(rxn)
                    for r,di in zip(rxn,d):
                        r = '<sos> '+r.strip()+' <eos>'
                        r = r.split(" ")
                        di['tgt'] = r
                        d_tgt.append(di)

        return d

    def len_vocab(self):

        path = "..\\data"
        len_vocab = []
        types = ["*.src","*.tgt"] # the tuple of file types
        for t in types:
            for v in glob.glob(os.path.join(path,t)):
                with open(v, "r") as inp:
                    lines = inp.readlines()
                    len_vocab.append(lines)

        return len_vocab

    def char_generator(self, vocab_src, vocab_tgt):
        import re
        vocab = ['<sos>','<eos>']+vocab_src+vocab_tgt
        chars = ""
        for v in vocab:
            char = re.findall(re.compile("([^\s]+)"), v)
            if f"{char[0]} " not in chars:
                chars = chars+" "+char[0]

        return chars

    def iterator(self, chars):

        stoi = { ch:i for i,ch in enumerate(chars.split(" "))}
        itos = { i:ch for i,ch in enumerate(chars.split(" ")) }

        return stoi,itos

    def encode_data(self, vocab_src, vocab_tgt, data):

        chars = self.char_generator(vocab_src, vocab_tgt)
        stoi,itos = self.iterator(chars)
        encode = lambda s: [stoi[c] for c in s ] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        d_iter_list = []
        for d in data:
            d_iter = {}
            for k,v in d.items():
                if k == 'src':
                    encoded_data = torch.tensor(encode(d[k]), dtype=torch.long)
                    d_iter[k] = encoded_data
                if k == 'tgt':
                    encoded_data = torch.tensor(encode(d[k]), dtype=torch.long)
                    d_iter[k] = encoded_data
            d_iter_list.append(d_iter)
        
        return d_iter_list

    def get_batch(self, data, batch_size):

        x_list = []
        y_list = []
        data_src_base = []
        data_tgt_base = []
        count = 0
        for d in data:
            data_src_base.append(d['src'])
            data_tgt_base.append(d['tgt'])
            count += 1

            if count == batch_size:      
                x = pad_sequence([data_src_base[0], data_src_base[1], data_src_base[2], data_src_base[3]], batch_first = True)
                y = pad_sequence([data_tgt_base[0], data_tgt_base[1], data_tgt_base[2], data_tgt_base[3]], batch_first = True)

                if x.size(dim=1) > y.size(dim=1):
                    y = F.pad(y, pad=(0, x.size(dim=1) - y.size(dim=1), 0, 0))
                elif x.size(dim=1) < y.size(dim=1):
                    x = F.pad(y, pad=(0, y.size(dim=1) - x.size(dim=1), 0, 0))

                x_list.append(x)
                y_list.append(y)
                data_src_base = []
                data_tgt_base = []                   
                count = 0

        return x_list,y_list

