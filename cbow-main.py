# coding: utf-8
import numpy as np
import codecs
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers, Variable
import chainer.functions as F
import argparse
from gensim import corpora, matutils
from gensim.models import word2vec
import time
import math
from net import CBOW
import util
import nltk.translate.bleu_score
import scipy.spatial.distance

"""
    Code for Continuous Bag-Of-Words
"""

def save_model(model, model_name):
    """ modelを保存
    """
    print ('save the model')
    serializers.save_npz('./models/' + model_name + '.model', model)

def save_optimizer(optimizer, model_name):
    """ optimzierを保存
    """
    print ('save the optimizer')
    serializers.save_npz('./models/' + model_name + '.state', optimizer)


def argument_parser():
    """ オプション設定
    """

    # デフォルト値の設定
    def_train = False
    def_test = False
    def_gpu = False
    def_is_debug_mode = False
    def_src = ""
    def_context_window = 4
    def_model = "cbow"

    # Model parameter
    def_vocab = 5000
    def_embed = 100
    def_hidden = 100

    # Other parameter
    def_epoch = 10
    def_batchsize = 40
    def_grad_clip = 5

    def_src_word = ""

    # 引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('src',
                        type=str,
                        default=def_src,
                        help='input file')
    parser.add_argument('--train',
                        dest="train",
                        action="store_true",
                        default=def_train,
                        help="if set, run train mode")
    parser.add_argument('--test',
                        dest="test",
                        action="store_true",
                        default=def_test,
                        help="if set, run test mode")
    parser.add_argument('-d',
                        '--debug',
                        dest="is_debug_mode",
                        action="store_true",
                        default=def_is_debug_mode,
                        help="if set, run train with debug mode")
    parser.add_argument('--use-gpu  ',
                        dest='use_gpu',
                        action="store_true",
                        default=def_gpu,
                        help='use gpu')
    parser.add_argument('--model ',
                        dest='model',
                        type=str,
                        default=def_model,
                        help='model file name to save')
    parser.add_argument('--vocab ',
                        dest='vocab',
                        type=int,
                        default=def_vocab,
                        help='vocabulary size')
    parser.add_argument('--embed',
                        dest='embed',
                        type=int,
                        default=def_embed,
                        help='embedding layer size')
    parser.add_argument('--hidden',
                        dest='hidden',
                        type=int,
                        default=def_hidden,
                        help='hidden layer size')

    parser.add_argument('--epoch',
                        dest='epoch',
                        type=int,
                        default=def_epoch,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize',
                        dest='batchsize'  ,
                        type=int,
                        default=def_batchsize,
                        help='learning minibatch size')
    parser.add_argument('--gclip',
                        dest='grad_clip'  ,
                        type=int,
                        default=def_grad_clip,
                        help='threshold of gradiation clipping')

    parser.add_argument('--word',
                        dest='src_word'  ,
                        type=str,
                        default=def_src_word,
                        help='a word that you want to similar words')

    return parser.parse_args()


def forward_one_step(model,
                     src_batch,
                     src_vocab2id,
                     is_train,
                     xp):
    """ 損失を計算
    """
    generation_limit = 256
    batch_size = len(src_batch)

    hyp_batch = [[] for _ in range(batch_size)]


    # Train
    if is_train:

        loss = Variable(xp.asarray(xp.zeros(()), dtype=xp.float32))

        src_batch =  [ [src_vocab2id["<s>"]] + src for src in src_batch]


        src_batch = xp.asarray(src_batch, dtype=xp.int32).T # 転置
        N = 2
        for i, t_batch in enumerate(src_batch[N:len(src_batch)-N]):
            index = i + N
            t = Variable(t_batch) #target

            context = []
            for offset in range(i, index+N+1):
                if offset == index:
                    continue
                context.append(Variable(src_batch[offset]))

            y = model(context)

            loss += F.softmax_cross_entropy(y, t)
            output = cuda.to_cpu(y.data.argmax(1))

            for k in range(batch_size):
                hyp_batch[k].append(output[k])


        return hyp_batch, loss # 予測結果と損失を返す

    # Test
    else:
        while len(hyp_batch[0]) < generation_limit:
            y = model.decode(t)
            output = cuda.to_cpu(y.data.argmax(1))
            t = Variable(xp.asarray(output, dtype=xp.int32))

            for k in range(batch_size):
                hyp_batch[k].append(output[k])
            if all(hyp_batch[k][-1] == trg_vocab2id['</s>'] for k in range(batch_size)):
                break
        return hyp_batch # 予測結果を返す


def train(args):
    """ 学習を行うメソッド
    """

    # オプションの値をメソッド内の変数に渡す
    vocab_size  = args.vocab      # 語彙数
    embed_size  = args.embed      # embeddingの次元数
    hidden_size = args.hidden     # 隠れ層のユニット数
    batchsize   = args.batchsize  # バッチサイズ
    n_epoch     = args.epoch      # エポック数(パラメータ更新回数)
    grad_clip   = args.grad_clip  # gradiation clip


    # 学習データの読み込み
    # Source
    print 'loading training data...'
    src_dataset, src_vocab2id, src_id2vocab = util.load_src_data(args.src, vocab_size)

    sample_size = len(src_dataset)

    # debug modeの時, パラメータの確認
    if args.is_debug_mode:
        print "[PARAMETERS]"
        print 'vocab size:', vocab_size
        print 'embed size:', embed_size
        print 'hidden size:', hidden_size

        print 'mini batch size:', batchsize
        print 'epoch:', n_epoch
        print 'grad clip threshold:', grad_clip
        print
        print 'sample size:', sample_size
        print

    # モデルの定義
    model = CBOW(vocab_size, embed_size)

    # GPUを使うかどうか
    if args.use_gpu:
        cuda.check_cuda_available()
        cuda.get_device(1).use()
        model.to_gpu()
    xp = cuda.cupy if args.use_gpu else np #args.gpu <= 0: use cpu, otherwise: use gpu


    N = sample_size
    # Setup optimizer
    optimizer = optimizers.AdaGrad(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))


    # 学習の始まり
    for epoch in range(n_epoch):
        print 'epoch:', epoch, '/', n_epoch

        # training
        perm = np.random.permutation(N) #ランダムな整数列リストを取得
        sum_train_loss = 0.0
        cur_log_perp = xp.zeros(())

        j = 0
        for i in six.moves.range(0, N, batchsize):

            #perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
            src_batch = src_dataset[perm[i:i + batchsize]]

            # 各バッチ内のサイズを統一させる
            src_batch = util.fill_batch(src_batch, src_vocab2id['</s>'])

            model.zerograds() # 重みを初期化

            # 損失を計算
            hyp_batch, loss = forward_one_step(model,
                                               src_batch,
                                               src_vocab2id,
                                               args.train,
                                               xp) # is_train
            cur_log_perp += loss.data
            sum_train_loss  += float(cuda.to_cpu(loss.data)) * len(src_batch)   # 平均誤差計算用

            loss.backward() # Backpropagation
            optimizer.update() # 重みを更新

            # デバッグ時だけ表示
            if args.is_debug_mode:
                for k, hyp in enumerate(hyp_batch):
                    print 'epoch: ', epoch, ', sample:', batchsize * j + k
                    _src = [src_id2vocab[x] if src_id2vocab[x] != "</s>" else "" for x in src_batch[k]]
                    _hyp = [src_id2vocab[x] if src_id2vocab[x] != "</s>" else "" for x in hyp]
                    print 'src:', ' '.join( _src )
                    print 'hyp:', ' '.join( _hyp )
                    print '=============================================='

            j += 1

        # 単語wordのembeddingを取得
        #embedding_list = model.get_embedding(Variable(xp.asarray([src_vocab2id[args.src_word]], dtype=xp.int32)))
        #print args.src_word, embedding_list.data

        print('train mean loss={}'.format(sum_train_loss / N)) #平均誤差
        print('training perplexity={}'.format(math.exp(float(cur_log_perp) / N))) #perplexity

        #モデルの途中経過を保存
        print 'saving model....'
        prefix = './model/' + args.model + '.%03.d' % (epoch + 1)
        util.save_vocab(prefix + '.srcvocab', src_id2vocab)
        model.save_spec(prefix + '.spec')
        serializers.save_hdf5(prefix + '.weights', model)

        sys.stdout.flush()

def test(args):
    """ 予測を行うメソッド
    """

    batchsize   = args.batchsize  # バッチサイズ

    # 語彙辞書の読込
    src_vocab2id, src_id2vocab, vocab_size = util.load_vocab(args.model + ".srcvocab")

    # モデルの読込
    model = CBOW.load_spec(args.model + ".spec")

    # GPUを使うかどうか
    if args.use_gpu:
        cuda.check_cuda_available()
        cuda.get_device(1).use()
        model.to_gpu()

    xp = cuda.cupy if args.use_gpu else np # args.gpu <= 0: use cpu, otherwise: use gpu
    serializers.load_hdf5(args.model + ".weights", model)

    # Source sequence for test
    print 'loading source data for test...'
    # データセット読み込み
    test_src_dataset = util.load_test_src_data(args.src, src_vocab2id)

    generated = 0
    N = len(test_src_dataset) # テストの事例数

    # 単語wordのembeddingを取得
    src_embed = model.embed.W.data[src_vocab2id[args.src_word]]

    print "src word:", args.src_word
    print src_embed

    trg_embed_list = {}
    for _word, _id in src_vocab2id.items():
        trg_embed = model.embed.W.data[src_vocab2id[_word]]
        trg_embed_list[_word] = 1 - scipy.spatial.distance.cosine(src_embed, trg_embed)

    # 上位10件を表示
    for i, (word, sim) in enumerate(sorted(trg_embed_list.items(), key=lambda x:x[1], reverse=True)):
        print word, sim
        if i == 10:
            break

def main():
    args = argument_parser()

    if args.train:
        train(args)
    elif args.test:
        test(args)


if __name__ == "__main__":
    main()
