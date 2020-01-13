import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
# from memory_profiler import profile


class BiLSTM(torch.nn.Module):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, embedding_matrix):
        super(BiLSTM, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)

        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.word_embeddings.weights = nn.Parameter(et, requires_grad=False)

        self.dropout = 0.1
        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64, 1)


def forward(self, x):
    h_embedding = self.embedding(x)
    h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

    h_lstm, _ = self.lstm(h_embedding)
    avg_pool = torch.mean(h_lstm, 1)
    max_pool, _ = torch.max(h_lstm, 1)
    conc = torch.cat((avg_pool, max_pool), 1)
    conc = self.relu(self.linear(conc))
    conc = self.dropout(conc)
    out = self.out(conc)
    return out