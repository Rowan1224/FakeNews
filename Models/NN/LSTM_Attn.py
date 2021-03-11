# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, h):
        # h : Batch * timestep * dimension
        #         print('h', h.shape)
        x = self.u(h)
        # u(h) : Batch * timestep * att_dim
        # print('u(h)', x)

        # tan(x) : Batch * timestep * att_dim
        x = self.tanh(x)
        # print('tanh(x)', x)

        # softmax(x) : Batch * timestep * att_dim
        x = self.softmax(x)
        # print(x)
        # print('softmax(h)', x.shape,  h.shape)
        # Batch matrix multiplication
        output = x * h
        #         print('output ', output.shape)
        output = torch.sum(output, dim=1)
        #         print('output ', output.shape)
        return output


class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, embedding_matrix):
        super(AttentionModel, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.word_embeddings.weights = nn.Parameter(et, requires_grad=False)
        self.lstm = nn.LSTM(embedding_length, hidden_size=hidden_size, batch_first=True,
                            dropout=0.5, num_layers=2, bidirectional=True)
        self.label = nn.Linear(hidden_size * 2, output_size)
        self.attn_module = Attention(hidden_size * 2)



    def forward(self, input_sentences, batch_size=None):

        input = self.word_embeddings(input_sentences)
        output, (final_hidden_state, final_cell_state) = self.lstm(input)  
        attn_output = self.attn_module(output)
        logits = self.label(attn_output)
        return logits
