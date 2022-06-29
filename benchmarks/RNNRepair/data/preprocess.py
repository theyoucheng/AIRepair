import torch
from torchtext import data
import joblib


import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer

from tqdm import tqdm 
np.random.seed(123)
torch.manual_seed(123)
########
### 
### data selection
###
########
def start_data_selection():

    comp = './toxic_data/data/' #/home/xiaofei/research/toxic/'
    
    TRAIN_DATA_FILE=f'{comp}train.csv'
    TEST_DATA_FILE=f'{comp}test.csv'
    
    embed_size = 50 # how big is each word vector
    max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 100 # max number of words in a comment to use
    
    
    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)
    
    list_sentences_train = train["comment_text"].fillna("_na_").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    
    list_sentences_test = test["comment_text"].fillna("_na_").values
    
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    
    new_train_index = []
    for i, str in tqdm(enumerate(list_sentences_train),total=len(list_sentences_train) ):
        if len(str)<200:
            new_train_index.append(i)
    new_train_index = np.asarray(new_train_index)
    new_train = list_sentences_train[new_train_index]
    new_y = np.sum(y, axis=1)[new_train_index]
    pos_indexes = np.where(new_y==0)[0]
    neg_indexes = np.where(new_y!=0)[0]
    
    pos_random_indexes = np.random.choice(np.arange(len(pos_indexes)), len(neg_indexes), replace=False)
    pos_train = new_train[pos_indexes][pos_random_indexes]
    pos_label = np.ones(len(pos_train), dtype=np.uint8)
    neg_train = new_train[neg_indexes]
    neg_label = np.zeros(len(neg_train), dtype=np.uint8)
    
    whole_train = np.concatenate([pos_train, neg_train], axis=0)
    whole_label = np.concatenate([pos_label, neg_label], axis=0)
    shuffle_idnexes = np.arange(len(whole_label))
    np.random.shuffle(shuffle_idnexes)
    final_train = whole_train[shuffle_idnexes]
    final_label = whole_label[shuffle_idnexes]
    
    
    temp_ = np.arange(len(final_label))
    train_indexes_ = np.random.choice(temp_, int(0.9*len(final_label)), replace=False)
    test_indexes_ = np.setdiff1d(temp_, train_indexes_)
    
    print(len(train_indexes_), len(test_indexes_))
    
    np.savez('./toxic_data/data/toxic_train.npz', train=final_train[train_indexes_], label=final_label[train_indexes_])
    
    
    np.savez('./toxic_data/data/toxic_test.npz', train=final_train[test_indexes_], label=final_label[test_indexes_])


########
### 
### data filter
###
########
def start_data_filter():
    import sys
    sys.path.append("../RNNRepair/use_cases/sentiment_analysis")
    from toxic_datasets import TOXIC
    
    try :
        TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language="en_core_web_sm")
    except :
        TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language="en")
    LABEL = data.LabelField(dtype=torch.float)
    
    
    train_data, test_data =TOXIC.splits(TEXT, LABEL, root='.', path="./toxic_data/data/")
    print(len(train_data), len(test_data))
    train_str = []
    
    test_str = []
    for example in train_data:
        train_str.append((example.label, example.text))
    
    
    
    for example in test_data:
        test_str.append((example.label, example.text))
    
    joblib.dump(train_str, './toxic_data/data_list/toxic_train.lst')
    joblib.dump(test_str, './toxic_data/data_list/toxic_test.lst')
    
    print('finish')

if not os.path.isfile("./toxic_data/data_list/toxic_train.lst"):
    os.makedirs("./toxic_data/data_list/",exist_ok=True)
    os.makedirs("./toxic_data/data/",exist_ok=True)
    
    print ("start preprocess")
    start_data_selection()
    start_data_filter()
    '''
    dac42938a7929c5b3d473b2f6a60141e  toxic_data/data_list/toxic_test.lst
    b1d3b6f6e25e0e61ffa4b0ed5f81b868  toxic_data/data_list/toxic_train.lst
    '''
else :
    print ("file exist in ./toxic_data/data_list/toxic_train.lst")