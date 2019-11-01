# _*_ coding: utf-8 _*_

import os
import sys
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.nn import functional as F
import numpy as np
# from torchtext import data
# from torchtext import datasets
# from torchtext.vocab import Vectors, GloVe
from keras_preprocessing import sequence,text

import string, re

from torch.utils.data import TensorDataset, DataLoader


def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """


    EMBEDDING_FILE = '/media/MyDrive/Project Fake news/Models/cc.bn.300.vec'


    # tokenize = lambda x: x.split()
    # TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    # LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    # train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    # LABEL.build_vocab(train_data)
    #
    # word_embeddings = TEXT.vocab.vectors
    # print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    # print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    # print ("Label Length: " + str(len(LABEL.vocab)))
    #
    # train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    # train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
    #
    # '''Alternatively we can also use the default configurations'''
    # # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    df = pd.read_csv("../Data/Corpus/AllDataTarget.csv")
    X = df["news"].values
    Y = df["label"].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

    # data preprocessing
    print(X[0])
    puncList = ["।", "”", "“", "’"]
    x = "".join(puncList)
    filterString = x + '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n০১২৩৪৫৬৭৮৯'
    tokenizer = text.Tokenizer(num_words=50000, filters=filterString, lower=False,)
    tokenizer.fit_on_texts(x_train)
    train_idx = tokenizer.texts_to_sequences(x_train)
    test_idx = tokenizer.texts_to_sequences(x_test)
    word_index = tokenizer.word_index

    embeddings_index = {}
    for i, line in enumerate(open(EMBEDDING_FILE)):
        val = line.split()
        embeddings_index[val[0]] = np.asarray(val[1:], dtype='float32')
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    x_train = sequence.pad_sequences(train_idx, maxlen=32, padding='post', truncating='post')
    x_test = sequence.pad_sequences(test_idx, maxlen=32, padding='post', truncating='post')

    # split_size = int(0.8 * len(x_train))
    # index_list = list(range(len(x_train)))
    # train_idx, valid_idx = index_list[:split_size], index_list[split_size:]
    # x_tr = torch.tensor(x_train[train_idx], dtype=torch.long)
    # y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
    # train = TensorDataset(x_tr, y_tr)
    # trainloader = DataLoader(train, batch_size=128)
    #
    # x_val = torch.tensor(x_train[valid_idx], dtype=torch.long)
    # y_val = torch.tensor(y_train[valid_idx], dtype=torch.float32)
    # valid = TensorDataset(x_val, y_val)
    # validloader = DataLoader(valid, batch_size=128)

    # split dataset to test and dev
    test_size = len(x_test)

    dev_size = (int)(test_size * 0.1)

    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]

    x_train = torch.tensor(x_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train = TensorDataset(x_train, y_train)
    train_iter = DataLoader(train, batch_size=32)

    x_test = torch.tensor(x_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    test = TensorDataset(x_test, y_test)
    test_iter = DataLoader(test, batch_size=32)

    x_dev = torch.tensor(x_dev, dtype=torch.long)
    y_dev = torch.tensor(y_dev, dtype=torch.float32)

    valid = TensorDataset(x_dev, y_dev)
    valid_iter = DataLoader(valid, batch_size=32)
    word_embeddings = embedding_matrix
    vocab_size = 50000


    return vocab_size, word_embeddings, train_iter, valid_iter, test_iter
