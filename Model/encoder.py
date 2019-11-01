import torch
from torch import nn
from torch.nn import functional as F
from models.attention import Attention

debug = False


def print_msg(*msg):
    if debug:
        print(msg)


class DocumentEncoderRNN(nn.Module):
    def __init__(self,
                 input_feauters,
                 rnn_units,
                 # max_seq_len,
                 pool_method,
                 encoder_type,
                 hidden_middle_val):
        super(DocumentEncoderRNN, self).__init__()

        # self.max_seq_len = max_seq_len
        self.emb_dim = input_feauters
        self.rnn_units = rnn_units
        self.pool_method = pool_method
        self.hidden_middle_val = hidden_middle_val * 2
        self.encoder_type = encoder_type

        self.encoder = encoder_type(self.emb_dim, self.rnn_units, bidirectional=True, batch_first=True)
        # self.hidden = self.init_hidden()

        if self.pool_method == 'attention':
            self.attention = Attention(self.rnn_units * 2)
        elif self.pool_method == 'relative_attention':
            self.attention = RelativeAttention(self.rnn_units * 2)

    def forward(self, matrix, mask=None):
        """
        Parameters
        ----------
        matrix : 4D matrix (dim: B * M * N * emb)
            B: Batch Size
            M: Sentences
            N: Words
            embd: Embedding dimension
        Returns
        -------
        Pooled output of size B * M * rnn_units
        """
        batch_size, max_doc_len, max_sent_len = matrix.size()

        print_msg('matrix', matrix.size())

        """ Reshape to flatten batches """
        self.encoder.flatten_parameters()
        if self.encoder_type == nn.LSTM:
            out, hidden = self.encoder(matrix, (self.init_hidden(batch_size), self.init_hidden(batch_size)))
        else:
            out, hidden = self.encoder(matrix, self.hidden[:, :matrix.size()[0], :].contiguous())
        print_msg('rnn out', out.size())

        """ Apply attention. Get the attention weights as extra output. """
        out, alphas, _ = self.pool(out, mask)

        # print_msg(out)
        print_msg('pooled out', out.size())

        return out, alphas

    def pool(self, context_vector, mask=None):
        print_msg('context', context_vector.size(), context_vector.permute(0, 2, 1).size())
        if self.pool_method == 'last_layer':
            return context_vector[:, -1, :]

        if self.pool_method == 'maxpool':
            print_msg('maxpool over time', self.max_seq_len)
            return F.max_pool1d(context_vector.permute(0, 2, 1), self.max_seq_len)

        if self.pool_method == 'attention' or self.pool_method == 'relative_attention':
            return self.attention(context_vector, mask)

    def init_hidden(self, seq_len):
        h = torch.zeros(2, seq_len, self.rnn_units)

        if torch.cuda.is_available():
            h = h.cuda()

        return h