import os
import random
import torch
from torch import nn
from d2l import torch as d2l
import torch.utils.data as data
import torchtext.vocab as vb

import sys
sys.path.append('..')
import sentiment_analyse_example.utils as utils


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = '../../data'

from tqdm import tqdm


def read_imdb(fold='train', data_root=DATA_ROOT+'/aclImdb'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, fold, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
        random.shuffle(data)
    return data


train_data, test_data = read_imdb('train'), read_imdb('test')


# 基于空格分词
def get_tokenized_imdb(data):
    """
    :param data: list of [string, label]
    :return:
    """
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


# 创建词典
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    # counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # return vocab.vocab(counter, min_freq=5)
    return d2l.Vocab(tokenized_data, min_freq=5)


vocab = get_vocab_imdb(train_data)


def preprocess_imdb(data, vocab):
    # 将每条评论通过截断或者补0，使得长度变成500
    max_l = 500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


batch_size = 8
train_set = data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_iter = data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = data.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X:', X)
    print('X: %s, y: %s' % (X.shape, y.shape))
    break

'batches: ', len(train_iter)


# 创建模型：双向循环神经网络
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional 设置为True即为双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # input形状 (批量大小，词数)。因为LSTM需要将序列长度(词数)作为第一维， 所以将输入转置后，再提取词特征
        # 输出形状为(词数，批量大小，词向量维度)
        embedding = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embedding，因此只返回最后一层的隐藏层在各时间步的隐藏状态
        # output形状(词数， 批量大小， 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embedding)  # output, (h, c)
        # 连接初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为(批量大小， 4*隐藏单元个数)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs


embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_hiddens)

# 加载预训练的词向量
glove_vocab = vb.GloVe(name='6B', dim=100, cache=os.path.join(DATA_ROOT, 'glove'))


def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    # out of vocabulary
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1

    if oov_count > 0:
        print("There are %d oov words." % (oov_count))
    return embed


net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.idx_to_token, glove_vocab))
net.embedding.weight.requires_grad = False # 直接加载预训练好的， 所以不需要更新它


# 训练并评估模型
lr, num_epochs = 0.01, 10
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


# 预测函数
def predict_sentiment(net, vocab, sentence):
    """sentence是词语列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'

predict_sentiment(net, vocab, ['The', 'heroine', 'is', 'a', 'bit', 'ugly'])
predict_sentiment(net, vocab, ['Poor', 'actors'])