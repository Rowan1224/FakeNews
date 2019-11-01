import torch
from torch import nn

debug = False


def print_msg(*msg):
    if debug:
        print(msg)


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = torch.Tensor([1e-10])

        if torch.cuda.is_available():
            self.epsilon = self.epsilon.cuda()


    def forward(self, h, mask):
        # b, t, dim = h.size()
        # h : Batch * timestep * dimension
        print_msg('h', h.shape, mask.shape)

        # u(h) : Batch * timestep * att_dim
        u_it = self.u(h)
        print_msg('u(h)', u_it.size())

        # tan(x) : Batch * timestep * att_dim
        u_it = self.tanh(u_it)

        # alpha = self.softmax(torch.matmul(u_it, self.v))
        # print_msg(alpha)
        # alpha = mask * alpha

        """ Direct softmax considers whole sequence. It contains padding. So manual"""
        # # softmax(x) : Batch * timestep * att_dim
        alpha = torch.exp(torch.matmul(u_it, self.v))
        print_msg(alpha)

        print_msg(type(mask), type(alpha), type(self.epsilon))
        print_msg(mask, alpha, self.epsilon)
        #print(mask.is_cuda, alpha.is_cuda, self.epsilon.is_cuda)

        if torch.cuda.is_available():
            alpha = mask * alpha + torch.Tensor([1e-10]).cuda()
        else:
            alpha = mask * alpha + torch.Tensor([1e-10])
        print_msg('after mask', alpha, alpha.size())
        #
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        print_msg(denominator_sum, denominator_sum.size())
        #
        alpha = mask * (alpha / denominator_sum)
        # alpha[alpha != alpha] = 0.  # Set nans to 0
        # print_msg('divide', alpha)

        # alpha[abs(alpha) == float('Inf')] = 0.  # Set infs to 0
        # print_msg(alpha)

        # Batch matrix multiplication
        output = h * alpha.unsqueeze(2)
        print_msg('output ', output.shape)

        output = torch.sum(output, dim=1)
        print_msg('output ', output.shape)

        return output, alpha, h


if __name__ == '__main__':
    # debug = True

    """ Test """
    data = [
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 2, 3],
            [0, 0, 0, 0, 3, 2, 1, 2, 4],
            [0, 0, 3, 2, 3, 2, 1, 2, 4]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 4, 2, 2, 1],
         [0, 0, 0, 0, 0, 0, 1, 2, 3],
         [0, 0, 0, 0, 3, 2, 1, 2, 4],
         [0, 0, 3, 2, 3, 2, 1, 2, 4], ]
    ]

    data = torch.LongTensor(data)
    print(data)
    print(data.size())

    mask = (data > 0).float()
    print(mask)

    doc_mask = torch.sum(mask, dim=-1)
    print_msg(doc_mask)
    doc_mask = (doc_mask > 0).float()
    print_msg(doc_mask)

    emb = torch.nn.Embedding(5, 3)
    lstm = torch.nn.LSTM(3, 4)
    attn = Attention(4)

    x = emb(data)
    print(x.size())

    x, _ = lstm(x.view(-1, 5, 3))
    print(x.size())

    x, attn_weights = attn(x, mask)
    print(x.size(), attn_weights.size())
    print(attn_weights)