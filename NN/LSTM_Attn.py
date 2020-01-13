# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


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
                            dropout=0.9, num_layers=1)
        self.label = nn.Linear(hidden_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        # if batch_size is None:
        # 	h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        # 	c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        # else:
        # 	h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # 	c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits
