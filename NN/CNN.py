# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNN(nn.Module):
	def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, weights):
		super(CNN, self).__init__()

		"""
		Arguments
		---------
		batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
		out_channels : Number of output channels after convolution operation performed on the input matrix
		kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
		keep_probab : Probability of retaining an activation node during dropout operation
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embedding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
		--------
		
		"""
		self.batch_size = batch_size
		self.output_size = output_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_heights = kernel_heights
		self.stride = stride
		self.padding = padding
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		weights = torch.tensor(weights, dtype=torch.float32)
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
		self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
		self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
		self.conv4 = nn.Conv2d(in_channels, out_channels, (kernel_heights[3], embedding_length), stride, padding)
		self.dropout = nn.Dropout(keep_probab)
		self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)

	def conv_block(self, input, conv_layer):
		conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
		activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
		max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)

		return max_out


	def forward(self, input_sentences, batch_size=None):

		input = self.word_embeddings(input_sentences)
		# input = input.permute(1, 0, 2)
		# input.size() = (batch_size, num_seq, embedding_length)
		input = input.unsqueeze(1)
		# input.size() = (batch_size, 1, num_seq, embedding_length)
		max_out1 = self.conv_block(input, self.conv1)
		max_out2 = self.conv_block(input, self.conv2)
		max_out3 = self.conv_block(input, self.conv3)
		max_out4 = self.conv_block(input, self.conv4)
		#
		all_out = torch.cat((max_out1, max_out2, max_out3, max_out4), 1)


		# all_out.size() = (batch_size, num_kernels*out_channels)
		fc_in = self.dropout(all_out)
		# fc_in.size()) = (batch_size, num_kernels*out_channels)
		logits = self.label(fc_in)

		return logits