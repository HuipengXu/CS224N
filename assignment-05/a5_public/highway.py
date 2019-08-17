#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import LOGLEVEL
import logging

logging.basicConfig(level=LOGLEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HighWay(nn.Module):

    def __init__(self, e_word:int):
        """
        :param e_word: dimension of word embedding
        """
        super(HighWay, self).__init__()
        self.proj = nn.Linear(e_word, e_word)
        self.gate = nn.Linear(e_word, e_word)

    def forward(self, X_convout: torch.Tensor) -> torch.Tensor:
        """
        :param X_convout (bs, e_word): the results of convnet
        :return X_highway (bs, e_word): the word embedding produced by character-based convnet
        """
        x_proj = F.relu(self.proj(X_convout))
        logger.debug('the shape of x-proj is: {shape}'.format(shape=str(x_proj.size())))
        x_gate = torch.sigmoid(self.gate(X_convout))
        logger.debug('the shape of x-gate is: {shape}'.format(shape=str(x_gate.size())))
        x_highway = x_gate * x_proj + (1 - x_gate) * X_convout
        logger.debug('the shape of x-highway is: {shape}'.format(shape=str(x_highway.size())))
        return x_highway


if __name__ == '__main__':
    highway = HighWay(2)
    proj_weight = torch.arange(10, 14).view(2, 2).float()
    proj_bias = torch.arange(20, 22).view(1, 2).float()
    gate_weight = torch.arange(5, 9).view(2, 2).float()
    gate_bias = torch.arange(9, 11).view(1, 2).float()
    highway.proj.weight = nn.Parameter(proj_weight, requires_grad=True)
    highway.proj.bias = nn.Parameter(proj_bias, requires_grad=True)
    highway.gate.weight = nn.Parameter(gate_weight, requires_grad=True)
    highway.gate.bias = nn.Parameter(gate_bias, requires_grad=True)
    X_convout = torch.randn((5, 2))
    x_proj = F.relu(X_convout @ proj_weight.t() + proj_bias)
    x_gate = F.sigmoid(X_convout @ gate_weight.t() + gate_bias)
    true_x_highway = x_gate * x_proj + (1 - x_gate) * X_convout
    x_highway = highway(X_convout)
    logger.debug('true x highway shape is: {shape}'.format(shape=true_x_highway.size()))
    assert np.allclose(true_x_highway.detach().numpy(), x_highway.detach().numpy())

### END YOUR CODE 

