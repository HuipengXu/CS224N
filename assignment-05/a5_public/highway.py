#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighWay(nn.Module):

    def __init__(self, e_word:int, dropout_rate:int=0.2):
        """
        :param e_word: dimension of word embedding
        :param dropout_rate: Dropout Probability, for X_highway
        """
        super(HighWay, self).__init__()
        self.proj = nn.Linear(e_word, e_word)
        self.gate = nn.Linear(e_word, e_word)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X_convout: torch.Tensor):
        """
        :param X_convout: the results of convnet
        :return X_highway (torch.Tensor): the word embedding produced by character-based convnet
        """
        x_proj = F.relu(self.proj(X_convout))
        x_gate = F.sigmoid(self.gate(X_convout))
        x_highway = x_gate * x_proj + (1 - x_gate) * X_convout
        return x_highway

### END YOUR CODE 

