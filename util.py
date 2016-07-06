# -*- coding: utf-8 -*-

import sys
import numpy as np
from collections import defaultdict
from scipy import stats

"""
自分で作成したTOOLとか
"""

# input data
def load_src_data(fname, vocab_size):

    """
    ソースファイルを読み込み
    数値データとシンボルデータを分けてデータを入力
    """

    print fname

    sentence_list = []
    word_freq = defaultdict(lambda: 0) # 各シンボルの出現回数計算用

    with open(fname, "r") as f:
        # ファイルを一行ずつ渡す
        for line in f:

            line = line.strip()

            # 空白の行は無視
            if len(line) == 0:
                continue

            # 単語分割
            word_list = line.split()
            sentence_list.append(word_list)

            # 各シンボルの出現回数を数える
            for word in word_list:
                word_freq[word] += 1


    # 単語-ID、ID-単語 辞書を作成
    vocab2id = defaultdict(lambda: 0)
    vocab2id['<unk>'] = 0
    vocab2id['<s>'] = 1
    vocab2id['</s>'] = 2

    id2vocab = [""] * vocab_size
    id2vocab[0] = '<unk>'
    id2vocab[1] = '<s>'
    id2vocab[2] = '</s>'

    # 辞書を作成
    for i, (word, count) in zip(range(vocab_size - 3), sorted(word_freq.items(), key=lambda x:-x[1])):
        vocab2id[word] += i + 3
        id2vocab[i + 3]  += word

    # id化したデータセット
    dataset = [ [ vocab2id.get(word, vocab2id["<unk>"]) for word in word_list ] for word_list in sentence_list ]

    print 'dataset size', len(dataset)
    print 'symbol vocab size:', len(vocab2id)
    print 'symbol vocab size(actual):', len(word_freq)
    print

    return np.array(dataset), vocab2id, id2vocab

# input data
def load_test_src_data(fname, vocab2id):

    """
    テストデータ用の読み込み用
    学習データのvocab2id辞書を基に、単語列をid列に変換
    """

    print fname

    sentence_list = []

    with open(fname, "r") as f:
        # ファイルを一行ずつ渡す
        for line in f:
            line = line.strip()

            # 空白の行は無視
            if len(line) == 0:
                continue

            # 単語分割
            word_list = line.split()
            sentence_list.append(word_list)


    # id化したデータセット
    dataset = [ [ vocab2id.get(word, vocab2id["<unk>"]) for word in word_list ] for word_list in sentence_list ]

    print 'dataset size', len(dataset)
    print

    return np.array(dataset)

# バッチ内の全てのベクトルを同じ次元に揃える
def fill_batch(batch, token_id):
    max_len = max(len(x) for x in batch)

    filled_batch = []
    for x in batch:
        if len(x) < max_len:
            padding = [ token_id for _ in range(max_len - len(x))]
            filled_batch.append(x + padding)
        else:
            filled_batch.append(x)
    return filled_batch

def save_vocab(filename, id2vocab):
    """ ファイルに語彙辞書を保存
    """
    with open(filename, 'w') as fp:
        print >> fp, len(id2vocab)
        for i in range(len(id2vocab)):
            print >> fp, id2vocab[i]

def load_vocab(filename):
    """ ファイルから語彙辞書を読込
    """
    with open(filename) as fp:
        vocab_size = int(next(fp))
        vocab2id = defaultdict(lambda: 0)
        id2vocab = [""] * vocab_size
        for i in range(vocab_size):
            s = next(fp).strip()
            if s:
                vocab2id[s] = i
                id2vocab[i] = s

    return vocab2id, id2vocab, vocab_size
