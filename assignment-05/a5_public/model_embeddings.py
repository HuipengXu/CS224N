#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import HighWay
from utils import LOGLEVEL

# End "do not change"

import logging

logging.basicConfig(level=LOGLEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        self.embed_size = embed_size
        pad_token_ind = vocab.char2id['<pad>']
        self.char_embedding = nn.Embedding(len(vocab.char2id), 50, padding_idx=pad_token_ind)
        self.cnn = CNN(self.embed_size)
        self.highway = HighWay(self.embed_size)
        self.dropout = nn.Dropout(p=0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        char_embedding = self.char_embedding(input)  # (sentence_length, bs, max_word_length, e_char=50)
        sentences = []
        for sent in char_embedding:
            x_convout = self.cnn(sent.transpose(1, 2))
            logger.debug('the shape of x convout is: {shape}'.format(shape=x_convout.size()))
            x_highway = self.highway(x_convout)
            logger.debug('the shape of x highway is: {shape}'.format(shape=x_highway.size()))
            x_word_emb = self.dropout(x_highway)
            logger.debug('the shape of x word_emb is: {shape}'.format(shape=x_word_emb.size()))
            sentences.append(x_word_emb)
        return torch.stack(sentences)

        ### END YOUR CODE

