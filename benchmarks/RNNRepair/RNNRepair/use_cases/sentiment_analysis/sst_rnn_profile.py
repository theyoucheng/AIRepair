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
import joblib
from torchtext import data
from torchtext import datasets

from ...profile_abs import  Profiling

from ...utils import get_project_root
from ...utils import sst_data_path

from .sst_datasets import SST, SST_LST
from .my_torch_iterator import MyBucketIterator
from .nlp_model import RNN, epoch_time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

use_large = True
g_nlp_token="glove.840B.300d" if use_large else "glove.6B.100d"
g_nlp_dim=300  if use_large else 100

# torch.backends.cudnn.deterinistic = True
# torch.backends.cudnn.benchmark= False

class SSTClassifier(Profiling):
    def __init__(self, rnn_type, save_dir, epoch = 40, train_default = True, dataset_path = None, seed = 0):
        # Classifier

        super().__init__(save_dir)
          # the size of vocabulary = 437938
        self.seed = seed
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

        if save_dir is not None:
            self.model_path = os.path.join(self.model_dir, rnn_type + '_' + str(epoch) + '.ckpt')
        self.sen_path = sst_data_path if dataset_path is None else dataset_path

        try:
            self.TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language="en_core_web_sm")
        except :
            self.TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language="en")
        self.LABEL = data.LabelField(dtype=torch.float)

        self.BATCH_SIZE = 256
        self.MAX_VOCAB_SIZE = 17197

        if train_default:
            self.train(dataset=None, is_load=True, saved_path=self.model_path)
    def load_default_data(self):
        train_lst = joblib.load(os.path.join(self.sen_path, 'sst_train.lst'))
        test_lst = joblib.load(os.path.join(self.sen_path, 'sst_test.lst'))
        return train_lst, test_lst

    def train(self, dataset = None, is_load = False, saved_path = None):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        if dataset is None:
            train_lst = joblib.load(os.path.join(self.sen_path, 'sst_train.lst'))
            test_lst = joblib.load(os.path.join(self.sen_path, 'sst_test.lst'))
        else:
            train_lst, test_lst = dataset

        para = {'npy/train': train_lst, 'npy/test': test_lst}

        train_data, test_data = SST_LST.splits(self.TEXT, self.LABEL, root='npy', **para)
        print(len(train_data), len(test_data))
        self.TEXT.build_vocab(train_data,
                              vectors_cache=os.path.expanduser('~/.torch/.vector_cache'),
                              max_size=self.MAX_VOCAB_SIZE,
                              vectors=g_nlp_token,
                              unk_init=torch.Tensor.normal_)
        self.LABEL.build_vocab(train_data)
        train_iterator, test_iterator = MyBucketIterator.splits(
            (train_data, test_data),
            batch_size=self.BATCH_SIZE,
            sort_within_batch=False,
            random_state = random.getstate(),
            device=device)
        INPUT_DIM = len(self.TEXT.vocab)
        PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        model = RNN(INPUT_DIM, self.EMBEDDING_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, 1, False, self.DROPOUT,
                    PAD_IDX, self.rnn_type)

        pretrained_embeddings = self.TEXT.vocab.vectors


        print(pretrained_embeddings.shape, flush=True)

        model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = self.TEXT.vocab.stoi[self.TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(self.EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)


        if is_load == True and os.path.exists(saved_path):
            model.load_state_dict(torch.load(saved_path, map_location=device))
            model.to(device)
        else:
            import torch.optim as optim

            optimizer = optim.Adam(model.parameters())
            criterion = nn.BCEWithLogitsLoss()
            model = model.to(device)
            criterion = criterion.to(device)

            for epoch in range(self.n_epochs):
                # print('Start !!', epoch)
                start_time = time.time()
                train_loss, train_acc = self.m_train(model, train_iterator, optimizer, criterion)
                valid_loss, valid_acc = self.evaluate(model, test_iterator, criterion)
                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

            test_loss, test_acc = self.evaluate(model, test_iterator, criterion)
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
            if saved_path is not None:
                torch.save(model.state_dict(), saved_path)
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.model = model
        return model


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



    def eval_test(self, model, x_test, save_text=None, save_name = '', need_loss = False):

        para = {'train': None, 'npy/test': x_test }
        test_data = SST_LST.splits(self.TEXT, self.LABEL, root='npy', **para)

        test_iterator = MyBucketIterator.splits(
            (test_data),
            check_train=False,
            batch_size=self.BATCH_SIZE,
            sort_within_batch=False,
            device=device)


        test_iterator = test_iterator[0]
        saved = False if save_text is None else True

        x =  self.predict(model, test_iterator, save_text=saved, save_name=save_name, need_loss=need_loss)
        # if need_loss:
            # return x[0], x[4], x[5]
        # else:
            # return  x[0], x[4]
        return x 


    def predict(self, model, iterator, save_text=False, save_name=None, need_loss = False): # n * 28 * 28
        # assert np_data is None ,"np_data should be None and reuse The self.train_data "

        model.eval()
        # Test the model
        # dense_list, state_vec_softmax_list, state_vec_list ,state_vec_len_list= [],[],[],[]

        final_labels, state_labels,  state_softmax, state_vec_list, truth_labels, embedding =[], [], [], [], [], []
        text_saved = []
        epoch_loss = []
        if need_loss:
            criterion = nn.BCEWithLogitsLoss()
            criterion = criterion.to(device)
        with torch.no_grad():

            for i, batch in enumerate(iterator):
                text, text_lengths = batch.text

                label = batch.label

                dense,state_vec,state_vec_len =model(text, text_lengths)


                dense = dense.squeeze_(1)
                if need_loss:


                    for j in range(len(batch)):
                        loss = criterion(dense[j:j+1], batch.label[j:j+1])
                        epoch_loss.append(loss.item())


                dense_1dim = torch.sigmoid(dense)
                results = torch.round(dense_1dim)

                sv_softmax = model.fc(state_vec)
                softmax_1dim = torch.sigmoid(sv_softmax)
                softmax =  SSTClassifier.ut_convert_1dim_to_2dim_aftersigmoid(softmax_1dim)



                re_np = results.cpu().numpy()
                sv_softmax_np = softmax.cpu().numpy()
                sv_vec_np = state_vec.cpu().numpy()
                truth_np = label.cpu().numpy()
                temp_data = batch.new_data
                text_np = text.cpu().numpy()
                text_np = np.transpose(text_np)


                for j in range(len(batch)):
                    final_labels.append(re_np[j])
                    state_softmax.append(sv_softmax_np[j][:state_vec_len[j]][:])
                    state_labels.append(np.argmax(sv_softmax_np[j][:state_vec_len[j]][:], axis=-1))
                    state_vec_list.append(sv_vec_np[j][:state_vec_len[j]][:])
                    truth_labels.append(truth_np[j])
                    text_saved.append((temp_data[j].label, temp_data[j].text))
                    embedding.append(text_np[j][:state_vec_len[j]])
        if save_text:
            import joblib
            print('Save texts', save_name)
            joblib.dump(text_saved, os.path.join(self.input_dir, save_name))
            joblib.dump(embedding, os.path.join(self.input_dir, save_name+'.embedd'))

        if need_loss:
            return np.array(final_labels), np.array(state_labels), np.array(state_softmax), np.array(state_vec_list), np.array(truth_labels), epoch_loss
        return np.array(final_labels), np.array(state_labels), np.array(state_softmax), np.array(state_vec_list), np.array(truth_labels)



    def do_profile(self, test=False, save_text=True):
        print('profile device', device, test)
        if test:
            return self.predict(self.model, self.test_iterator, save_text=save_text, save_name='test.texts')
        else:
            return self.predict(self.model, self.train_iterator, save_text=save_text, save_name='train.texts')


    @staticmethod
    def ut_convert_1dim_to_2dim_aftersigmoid(sigmoid_val):
        if len(sigmoid_val.shape)<=1:
            sigmoid_val=sigmoid_val.unsqueeze(-1)
        total= torch.ones_like(sigmoid_val)
        sigmoid_val_neg= total-sigmoid_val
        dim2_pred=torch.cat([sigmoid_val_neg,sigmoid_val],dim=-1)
        return dim2_pred