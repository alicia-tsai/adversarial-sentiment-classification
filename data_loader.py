#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:27:14 2018

@author: aliciatsai
"""

# If error `[E050] Can't find model 'en'`. 
# Use the following command to download `en`: `python3 -m spacy download en`

import random

from torchtext import data
from torchtext import datasets
import torch


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class DataLoader:
    
    def __init__(self):
        # set up fields
        self.TEXT = data.Field(tokenize='spacy', lower=True)
        self.LABEL = data.Field(dtype=torch.float)
        
        # make splits for data
        print('loading data...')
        self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)

        # smaller subsets
        self.small_train = None
        self.small_valid = None

    def small_train_valid(self):
        print('splitting data...')
        # only use 2500 examples
        train, disgard = self.train_data.split(random_state=random.seed(SEED), split_ratio=0.1)
        # train, valid is 1250
        train, valid = train.split(random_state=random.seed(SEED), split_ratio=0.5)
        
        # build the vocabulary embedding
        print('building vocabulary...')
        self.TEXT.build_vocab(train, vectors="glove.6B.100d")
        self.LABEL.build_vocab(train)

        # store smaller subsets
        self.small_train, self.small_valid = train, valid
        
        return train, valid

    def small_train_valid_iter(self, batch_size=64, device=None):
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not (self.small_train and self.small_valid):
            self.small_train_valid()
        # make iterator for splits
        train_iter, valid_iter = data.BucketIterator.splits((self.small_train, self.small_valid), batch_size=batch_size, device=device)

        return train_iter, valid_iter
