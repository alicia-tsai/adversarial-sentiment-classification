#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:27:14 2018

@author: aliciatsai
"""

# If error `[E050] Can't find model 'en'`. 
# Use the following command to download `en`: `python3 -m spacy download en`

import random
import dill

from torchtext import data
from torchtext import datasets
import torch


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class IMDB(datasets.IMDB):
    """Override `datasets.IMDB` methods to make it work for pickling data (save data)."""

    def __getstate__(self):
        """Override __getstate__ to make `dill` working."""
        return dict(self.__dict__)

    def __setstate__(self, state):
        """Override __setstate__ to make `dill` working."""
        self.__dict__.update(state)


class DataLoader:
    
    def __init__(self):
        # set up fields
        self.TEXT = None
        self.LABEL = None
        
        # set up whole data sets
        self.train_data = None
        self.test_data = None

        # set up large subsets
        self.large_train = None
        self.large_valid = None

        # set up smaller subsets
        self.small_train = None
        self.small_valid = None

    def build_data(self, save=True, train_outfile='train_data.pt', test_outfile='test_data.pt'):
        """Build data sets from `torchtext.datasets.IMDB`."""

        # set up fields
        self.TEXT = data.Field(tokenize='spacy', lower=True)
        # self.TEXT = data.Field(lower=True)
        self.LABEL = data.Field(dtype=torch.float)

        # load data from `torchtext.datasets`
        # 25,000 examples for train_data, test_data respectively
        print('building data...')
        self.train_data, self.test_data = IMDB.splits(self.TEXT, self.LABEL)

        if save:
            print('saving data at %s, %s' %(train_outfile, test_outfile))
            with open(train_outfile, 'wb') as file:
                dill.dump(self.train_data, file)

            with open(test_outfile, 'wb') as file:
                dill.dump(self.test_data, file)

    def load_data(self, train_outfile='train_data.pt', test_outfile='test_data.pt'):
        """Load training and testing data sets from file."""
        print('loading data...')
        with open(train_outfile, 'rb') as file:
            self.train_data = dill.load(file)

        with open(test_outfile, 'rb') as file:
            self.test_data = dill.load(file)

    def _build_vocab(self, train):
        # build the vocabulary embedding
        print('building vocabulary...')
        self.TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d")
        self.LABEL.build_vocab(train)

    def large_train_valid(self, split_ratio=0.8):
        if not (self.train_data and self.test_data):
            self.build_data()

        print('splitting data...')
        large_train, large_valid = self.train_data.split(random_state=random.seed(SEED), split_ratio=split_ratio)

        self.TEXT = large_train.fields['text']
        self.LABEL = large_train.fields['label']
        self._build_vocab(large_train)
        self.large_train, self.large_valid = large_train, large_valid  # store large subsets

        return large_train, large_valid

    def small_train_valid(self, split_ratio=0.5):
        if not (self.train_data and self.test_data):
            self.build_data()

        print('splitting data...')
        # only use 2500 examples
        small_train, disgard = self.train_data.split(random_state=random.seed(SEED), split_ratio=0.1)
        # default 1250 examples for train, valid respectively
        small_train, small_valid = small_train.split(random_state=random.seed(SEED), split_ratio=split_ratio)

        self.TEXT = small_train.fields['text']
        self.LABEL = small_train.fields['label']
        self._build_vocab(small_train)
        self.small_train, self.small_valid = small_train, small_valid  # store small subsets
        
        return small_train, small_valid

    def train_valid_iter(self, small_subsets=False, batch_size=64, device=None):
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # make iterator for splits
        if small_subsets:
            if not (self.small_train and self.small_valid):
                self.small_train_valid()
            train_iter, valid_iter = data.BucketIterator.splits((self.small_train, self.small_valid),
                                                                batch_size=batch_size, device=device)
        else:
            if not (self.large_train and self.large_valid):
                self.large_train_valid()
            train_iter, valid_iter = data.BucketIterator.splits((self.large_train, self.large_valid),
                                                                batch_size=batch_size, device=device)
        return train_iter, valid_iter

    def small_train_valid_iter(self, batch_size=64, device=None):

        return self.train_valid_iter(small_subsets=True, batch_size=batch_size, device=device)

    def large_train_valid_iter(self, batch_size=64, device=None):

        return self.train_valid_iter(small_subsets=False, batch_size=batch_size, device=device)