#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import logging
import time
from transformers import BertModel, T5Model
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
"""
基于pytorch的BERT语言模型
"""

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"
handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
logger.addHandler(handler)

def get_pad_mask(seq, pad_idx):
    '''
    seq = torch.tensor([
        [5, 12, 7, 0, 0],  # 第一个句子，实际长度为3，后面填充了2个0
        [9, 2, 4, 6, 0]   # 第二个句子，实际长度为4，后面填充了1个0
    ])
    pad_idx = 0

    # tensor([
    #   [True, True, True, False, False],
    #   [True, True, True, True, False]
    # ])
    '''
    # (batch_size, seq_len) -> (batch_size, 1, seq_len)
    # eg. (64, 10) -> (64, 1, 10)
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    '''
    示例：如果len_s=4，最终会得到：
    [[[ True, False, False, False],
      [ True,  True, False, False],
      [ True,  True,  True, False],
      [ True,  True,  True,  True]]]
    '''
    sz_b, len_s = seq.size()    # (batch_size, seq_len)  eg. (64, 10)
    # torch.tril() 函数会返回一个矩阵的下三角部分（包括对角线）
    subsequent_mask = torch.tril(torch.ones((1, len_s, len_s), device=seq.device)).bool()
    return subsequent_mask   # (1, seq_len, seq_len)  eg. (1, 10, 10)

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.encoder = BertModel.from_pretrained(r"D:\pretrain_models\bert-base-chinese")
        # self.pad_token_id = self.encoder.config.pad_token_id
        self.pad_token_id = vocab["[PAD]"]
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):   # x.shape:(batch_size, sen_len)
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)  _.shape: 元组 (1, batch_size, input_dim), (1, batch_size, input_dim)

        attention_mask = get_pad_mask(x, self.pad_token_id) & get_subsequent_mask(x)  # output shape:(batch_size, seq_len, seq_len)
        x = self.encoder(x, attention_mask=attention_mask)[0]    #output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            # (number, class_num), (number)
            # 交叉熵对输入有形状要求(number, class_num), (number)，需要view转变张量形状
            # (batch_size, sen_len, vocab_size) -> (batch_size * sen_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)   # (batch_size, sen_len, vocab_size)


class DataGenerator:
    def __init__(self, data_path, sample_length, window_size, config=None):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(r"D:\pretrain_models\bert-base-chinese")
        self.vocab = self.tokenizer.get_vocab()
        self.corpus = load_corpus(data_path)
        self.sample_length = sample_length
        self.window_size = window_size
        self.load()

    def load(self):
        self.data = []
        # segment_input_ids = []
        # segment_attention_mask = []
        # labels = []
        # 随机截取训练文本
        for i in range(self.sample_length):
            # x, y = build_sample(vocab, window_size, corpus)
            start = random.randint(0, len(self.corpus) - 1 - self.window_size)
            end = start + self.window_size
            window = self.corpus[start:end]
            target = self.corpus[start + 1:end + 1]  # 输入输出错开一位
            # print(window, target)
            # 将字转换成序号
            # 输入句子需要加mask处理
            input_ids = self.encode_sentence(window)
            # 输出句子不需要mask处理，直接和模型预测输出计算loss
            labels = self.encode_sentence(target)
            self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_ids = [self.vocab.get(char, self.vocab["[UNK]"]) for char in text]
        if padding:
            input_ids = self.padding(input_ids, self.vocab["[PAD]"])
        return input_ids

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.window_size]
        input_id += [pad_token] * (self.window_size - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, sample_length, batch_size, window_size, shuffle=True):
    dg = DataGenerator(data_path, sample_length, window_size)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl


# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，输入窗口和输出窗口错开一位
# def build_sample(vocab, window_size, corpus):
#     start = random.randint(0, len(corpus) - 1 - window_size)
#     end = start + window_size
#     window = corpus[start:end]
#     target = corpus[start + 1:end + 1]  #输入输出错开一位
#     # print(window, target)
#     x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
#     y = [vocab.get(word, vocab["<UNK>"]) for word in target]
#     return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
# def build_dataset(sample_length, vocab, window_size, corpus):
#     dataset_x = []
#     dataset_y = []
#     for i in range(sample_length):
#         x, y = build_sample(vocab, window_size, corpus)
#         dataset_x.append(x)
#         dataset_y.append(y)
#     return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])   # (1， seq_len)
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]         # 最后一个字的(vocab_size,) 概率分布
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 50        #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    # vocab = build_vocab("vocab.txt")       #建立字表
    # corpus = load_corpus(corpus_path)     #加载语料
    #加载训练数据
    train_data = load_data(corpus_path, train_sample, batch_size, window_size)
    vocab = train_data.dataset.vocab
    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)   #建立优化器
    logger.info("文本词表模型加载完毕，开始训练")
    logger.info(f"epoch num: {epoch_num}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"learning rate: {optim.param_groups[0]['lr']}")
    for epoch in range(epoch_num):
        model.train()
        logger.info("=========\nepoch %d begin" % (epoch + 1))
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            # x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            # if torch.cuda.is_available():
            #     x, y = x.cuda(), y.cuda()
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            optim.zero_grad()    #梯度归零
            input_ids, labels = batch_data
            loss = model(input_ids, labels)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        logger.info("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        logger.info(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        logger.info(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        # base_name = os.path.basename(corpus_path).replace("txt", "pth")
        # model_path = os.path.join("model", base_name)
        # 创建保存模型的目录
        if not os.path.isdir("model_output"):
            os.mkdir("model_output")
        model_path = os.path.join("model_output", "epoch_%d.pth" % (epoch + 1))
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", True)
