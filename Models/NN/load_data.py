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

import sys
sys.path.insert(1, '../Helper/')
import config



def load_dataset(test_sen=None):


    EMBEDDING_FILE = config.EMBEDDING_300

    df = pd.read_csv(config.DATA_PATH)
    X = df["content"].values
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
    for i, line in enumerate(open(EMBEDDING_FILE, encoding="utf-8")):
        val = line.split()
        embeddings_index[val[0]] = np.asarray(val[1:], dtype='float32')
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    x_train = sequence.pad_sequences(train_idx, maxlen=32, padding='post', truncating='post')
    x_test = sequence.pad_sequences(test_idx, maxlen=32, padding='post', truncating='post')

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
