from transformer_for_reactions.preprocess import Preprocess
from transformer_for_reactions.model import Model

import torch

process = Preprocess()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = process.get_data(["..\\data\\train_src.txt","..\\data\\train_tgt.txt"])
valid_data = process.get_data(["..\\data\\valid_src.txt","..\\data\\valid_tgt.txt"])
test_data = process.get_data(["..\\data\\test_src.txt","..\\data\\test_tgt.txt"])

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

vocab_src, vocab_tgt = process.len_vocab()
print(f"Unique tokens in source vocabulary: {len(vocab_src)}")
print(f"Unique tokens in target vocabulary: {len(vocab_tgt)}")

train_data_encoded = process.encode_data(vocab_src, vocab_tgt, train_data)
valid_data_encoded = process.encode_data(vocab_src, vocab_tgt, valid_data)
test_data_encoded = process.encode_data(vocab_src, vocab_tgt, test_data)

batch_size = 4
x_train, y_train = process.get_batch(train_data_encoded, batch_size)
x_valid, y_valid = process.get_batch(valid_data_encoded, batch_size)
x_test, y_test = process.get_batch(test_data_encoded, batch_size)

model = Model(len(vocab_src),len(vocab_tgt), 256, 256, 512, 2, 0.5, 0.5, 4)
model.model_run(x_train, y_train, x_valid, y_valid, N_EPOCHS=10)
model.test_run(x_test, y_test,vocab_src, vocab_tgt)