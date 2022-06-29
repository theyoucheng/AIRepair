#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:38:58 2019

@author: root
"""
import sys

# sys.path.append("../../")
import torch
import torch.nn as nn
import numpy as np
import time
import os
import random

from torchtext import data
from torchtext import datasets

from ...profile_abs import  Profiling
from .my_torch_iterator import MyBucketIterator

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


use_large = True
g_nlp_token="glove.840B.300d" if use_large else "glove.6B.100d"
g_nlp_dim=300  if use_large else 100

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, rnn_type):


        super().__init__()
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)
        else:
            self.rnn = nn.GRU(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim , output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        if self.rnn_type == 'lstm':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        if hidden.shape[0]<=1:
            hidden = hidden.squeeze_(0)
            hidden = self.dropout(hidden)#torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden),output,output_lengths

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
class IMDBClassifier(Profiling):
    def __init__(self, rnn_type, save_dir, epoch = 5, save_period = 0,  overwrite = False):
        # Classifier
        super().__init__(save_dir)
          # the size of vocabulary = 437938
        self.n_classes = 2
        self.EMBEDDING_DIM =g_nlp_dim  # embedding every word to a 100 dim vector
        self.HIDDEN_DIM = g_nlp_dim  # hidden layer size (default: 32)
        self.OUTPUT_DIM = 1
        self.BIDIRECTIONAL = False
        self.DROPOUT = 0.8
        self.channel = 1
        self.n_epochs = epoch
        # Internal
        self._data_loaded = False
        self._trained = False
        self.rnn_type = rnn_type
        self._period = save_period

        self.model_path = os.path.join(self.model_dir, rnn_type + '_' + str(epoch) + '.ckpt')

        try:
            self.TEXT = data.Field(tokenize='spacy', include_lengths=True , tokenizer_language="en_core_web_sm")
        except :
            self.TEXT = data.Field(tokenize='spacy', include_lengths=True , tokenizer_language="en")
        self.LABEL = data.LabelField(dtype=torch.float)

        print('Start split training and test data')
        start = time.time()
        self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT, self.LABEL, root=os.path.expanduser('~/.torch/.data'))

        print('End split and time used', time.time()-start)


        SEED = 1234
        # self.train_data, self.valid_data = self.train_data.split(random_state=random.seed(SEED))

        MAX_VOCAB_SIZE = 25_000

        self.TEXT.build_vocab(self.train_data,
                         vectors_cache=os.path.expanduser('~/.torch/.vector_cache'),
                         max_size=MAX_VOCAB_SIZE,
                         vectors=g_nlp_token,
                         unk_init=torch.Tensor.normal_)

        self.LABEL.build_vocab(self.train_data)
        self.BATCH_SIZE = 1024
        self.INPUT_DIM = len(self.TEXT.vocab)
        PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]
        self.model = RNN(self.INPUT_DIM, self.EMBEDDING_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, 1, False, self.DROPOUT, PAD_IDX, rnn_type )
        self.model = self.model.to(device)
        
        self.train_iterator, self.test_iterator = MyBucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.BATCH_SIZE,
            sort_within_batch=True,
            device=device)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'The model has {count_parameters(self.model):,} trainable parameters')

        pretrained_embeddings = self.TEXT.vocab.vectors

        print(pretrained_embeddings.shape, flush=True)

        self.model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = self.TEXT.vocab.stoi[self.TEXT.unk_token]

        self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.EMBEDDING_DIM)
        self.model.embedding.weight.data[PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)


        if os.path.exists(self.model_path):
            print('Load existing model')
            self.model.load_state_dict(torch.load(os.path.join(self.model_path), map_location=device))
            self.model.to(device)
        else:
            print('Training...')
            self.train()
    def binary_accuracy_2dim(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        assert torch.sum( torch.isclose(preds.sum(dim=-1), torch.ones(preds.shape[0]).to(device) ) ) == preds.shape[0]

        rounded_preds = torch.argmax(preds,dim=-1)
        # print(rounded_preds.type(), y.type())
        correct = (rounded_preds == y.long()).float()  # convert into float for division
        # acc = correct.sum() / len(correct)
        # return acc
        return correct.sum(), len(correct)

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc
    def evaluate(self, model, iterator, criterion):

        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            co = []
            to = []
            for batch in iterator:
                text, text_lengths = batch.text

                predictions,_,_ = model(text, text_lengths)
                predictions = predictions.squeeze(1)

                loss = criterion(predictions, batch.label)

                correct, total = self.binary_accuracy_2dim(self.ut_convert_1dim_to_2dim_aftersigmoid(torch.sigmoid(predictions)), batch.label)
                co.append(correct)
                to.append(total)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator), sum(co) / sum(to)
    def m_train(self, model, iterator, optimizer, criterion):

        epoch_loss = 0
        epoch_acc = 0

        model.train()
        co = []
        to = []
        for batch in iterator:
            optimizer.zero_grad()

            text, text_lengths = batch.text

            predictions,_,_ = model(text, text_lengths)
            predictions  = predictions.squeeze_(1)

            loss = criterion(predictions, batch.label)
            loss.backward()
            optimizer.step()
            # print(batch.label.type())
            correct, total = self.binary_accuracy_2dim(self.ut_convert_1dim_to_2dim_aftersigmoid(torch.sigmoid(predictions)), batch.label)
            epoch_loss += loss.item()
            co.append(correct)
            to.append(total)

        return epoch_loss / len(iterator), sum(co)/sum(to)

    def train(self):

        import torch.optim as optim

        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        self.model = self.model.to(device)
        criterion = criterion.to(device)

        best_valid_loss = float('inf')

        for epoch in range(self.n_epochs):
            # print('Start !!', epoch)
            start_time = time.time()
            train_loss, train_acc = self.m_train(self.model, self.train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate(self.model, self.test_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_path)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        test_loss, test_acc = self.evaluate(self.model, self.test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    def predict(self, iterator, save_text=False, save_name=None):  # n * 28 * 28
        # assert np_data is None ,"np_data should be None and reuse The self.train_data "

        xx = []
        self.model.eval()
        # Test the model
        dense_list, softmax_list, state_vec_list, state_vec_len_list = [], [], [], []

        with torch.no_grad():
            text_saved = []
            for i, batch in enumerate(iterator):
                text, text_lengths = batch.text

                label = batch.label

                dense, state_vec, state_vec_len = self.model(text, text_lengths)
                dense = dense.squeeze_(1)

                dense_1dim = torch.sigmoid(dense)
                results = torch.round(dense_1dim)

                sv_softmax = self.model.fc(state_vec)

                softmax_1dim = torch.sigmoid(sv_softmax)
                softmax = IMDBClassifier.ut_convert_1dim_to_2dim_aftersigmoid(softmax_1dim)

                dense_list.append(results.cpu().numpy())
                softmax_list.append(softmax.cpu().numpy())
                state_vec_list.append(state_vec.cpu().numpy())
                xx.append(label.cpu().numpy())
                # state_vec_len_list.append(state_vec_len)

                if save_text:
                    temp_data = batch.new_data
                    for d in temp_data:
                        text_saved.append((d.label == 'pos', d.text))
        if save_text:
            import joblib
            print('Save texts', save_name)
            text_dir = os.path.join(self.save_dir, '_texts')
            os.makedirs(text_dir, exist_ok=True)
            joblib.dump(text_saved, os.path.join(text_dir, save_name))

        return output_process(dense_list, softmax_list, state_vec_list, np.concatenate(xx, axis=0))

    def do_profile(self, test=False):
        print('profile device', device, test)
        if test:
            return self.predict(self.test_iterator, save_text=True, save_name='test.texts')
        else:
            return self.predict(self.train_iterator, save_text=True, save_name='train.texts')

    @staticmethod
    def ut_convert_1dim_to_2dim_aftersigmoid(sigmoid_val):
        if len(sigmoid_val.shape)<=1:
            sigmoid_val=sigmoid_val.unsqueeze(-1)
        total= torch.ones_like(sigmoid_val)
        sigmoid_val_neg= total-sigmoid_val
        dim2_pred=torch.cat([sigmoid_val_neg,sigmoid_val],dim=-1)
        return dim2_pred


if __name__ == '__main__':
    print('use_gpu: ', use_cuda)
    classifier = IMDBClassifier(rnn_type='lstm', save_dir='../../data/torch_lstm_imdb', epoch=1, save_period=0,
                                 overwrite=False)
    classifier.do_profile(test=True)