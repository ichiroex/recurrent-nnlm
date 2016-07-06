# -*- coding: utf-8 -*-
"""
Network Architecture of Neural Language Model
"""
from __future__ import print_function
import chainer
from chainer import cuda, Variable, Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

class RNNLM(Chain):

    def __init__(self,
                 vocab_size,
                 embed_size):

        super(RNNLM, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            l1 = L.LSTM(embed_size, embed_size),
            l2 = L.Linear(embed_size, vocab_size))

        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def __call__(self, x):
        h = self.embed(x)
        h = self.l1(h)
        y = self.l2(h)
        return y

    def reset_state(self):
        self.l1.reset_state()

    def get_embedding(self, x):
        return self.embed(x)

    def save_spec(self, filename):
        with open(filename, 'w') as fp:
            # パラメータを保存
            print(self.vocab_size, file=fp)
            print(self.embed_size, file=fp)

    @staticmethod
    def load_spec(filename):
        with open(filename) as fp:
            # specファイルからモデルのパラメータをロード
            vocab_size = int(next(fp))
            embed_size = int(next(fp))
            return RNNLM(vocab_size, embed_size)
