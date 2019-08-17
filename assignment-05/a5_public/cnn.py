#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import LOGLEVEL
import logging

logging.basicConfig(level=LOGLEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CNN(nn.Module):

    def __init__(self, e_word:int, e_char:int=50, m_word:int=21, kernel_size:int=5):
        """
        :param e_char: character embedding size
        :param e_word: the size of the final word embedding
        :param m_word: the length of word
        :param kernel_size: the size of filter
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(e_char, e_word, kernel_size)
        self.max_pool = nn.MaxPool1d(m_word-kernel_size+1)

    def forward(self, X_reshape:torch.Tensor) -> torch.Tensor:
        """
        :param X_reshape (bs, e_char, max_word_length): a tensor pass the character embedding lookup
        :return X_convout (bs, e_word): word embedding
        """
        x_conv = self.conv(X_reshape) # (bs, e_word, m_word-kernel_size+1)
        logger.debug('the shape of x-conv is: {shape}'.format(shape=x_conv.size()))
        x_convout = self.max_pool(F.relu(x_conv))
        logger.debug('the shape of x-convout is: {shape}'.format(shape=x_convout.size()))
        return x_convout.squeeze(-1)


if __name__ == '__main__':
    cnn = CNN(2, 4, 6, 3)
    conv_weight = torch.randn(4, 2, 3)
    conv_bias = torch.randn(4)
    cnn.conv.weight = nn.Parameter(conv_weight, requires_grad=True)
    cnn.conv.bias = nn.Parameter(conv_bias, requires_grad=True)
    X_reshape = torch.randn(5, 2, 6)
    x_conv = []
    for x in X_reshape:
        tmp = torch.zeros((4, 4))
        for i in range(6-3+1):
            tmp[:, i] = (conv_weight * x[:, i:i+3]).view(4, -1).sum(dim=1) + conv_bias
        x_conv.append(tmp)
    x_conv = torch.stack(x_conv) # (5, 4, 4)
    true_x_convout = torch.where(x_conv >= 0, x_conv, torch.zeros(1)).max(dim=2)[0] # relu + max-pool
    x_convout = cnn(X_reshape)
    assert np.allclose(true_x_convout.detach().numpy(), x_convout.detach().numpy())

### END YOUR CODE

