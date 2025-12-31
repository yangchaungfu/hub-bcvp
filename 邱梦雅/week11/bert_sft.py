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
from print_mask import plot_mask
import json

"""
基于pytorch的BERT语言模型实现sft式的seq2seq训练
训练数据问答对格式：<内容，摘要>
"""

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"
handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
logger.addHandler(handler)

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab, mask_plot=False):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.encoder = BertModel.from_pretrained(r"D:\pretrain_models\bert-base-chinese", return_dict=False, attn_implementation='eager')
        # self.pad_token_id = self.encoder.config.pad_token_id
        self.pad_token_id = vocab["[PAD]"]
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy
        self.mask_plot = mask_plot

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):   # x.shape:(batch_size, sen_len)
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)  _.shape: 元组 (1, batch_size, input_dim), (1, batch_size, input_dim)

        if y is not None:
            batch_size, seq_len = x.size()  # s1 + s2
            s2_max_length = y.size()[1]   # s2的长度
            s1_max_length = seq_len - s2_max_length   # s1的长度
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((batch_size, seq_len, seq_len)))  # output shape:(batch_size, seq_len, seq_len)
            # 输入文本s1可以完整看见输入的所有内容，看不到输出s2的所有内容
            mask[:, :s1_max_length, :s1_max_length] = 1  # 前 s1_max_length x s1_max_length 区域设为1，表示输入s1的所有token之间全可见
            if self.mask_plot:
                plot_mask(mask, s1_max_length)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.encoder(x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)
            # 预测输出只用s2部分（回答）计算loss
            y_pred = y_pred[:, -s2_max_length:, :]  # output shape:(batch_size, s2_max_length, vocab_size)
            # (number, class_num), (number)
            # 交叉熵对输入有形状要求(number, class_num), (number)，需要view转变张量形状
            # (batch_size, s2_max_length, vocab_size) -> (batch_size * s2_max_length, vocab_size)
            return self.loss(y_pred.contiguous().view(-1, y_pred.shape[-1]), y.reshape(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.encoder(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)   # (batch_size, sen_len, vocab_size)


class DataGenerator:
    def __init__(self, data_path, sample_length, s1_max_length, s2_max_length, config=None):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(r"D:\pretrain_models\bert-base-chinese")
        self.vocab = self.tokenizer.get_vocab()
        # self.corpus = load_corpus(data_path)
        self.sample_length = sample_length
        self.s1_max_length = s1_max_length   # 105
        self.s2_max_length = s2_max_length   # 20
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                # 输入序列编码，不作特殊处理（内容文本，问答对的问题部分）
                input_seq = self.encode_sentence(content, self.s1_max_length, False, False)  # 输入序列
                # decoder输入编码，加开始符号（内容标题，问答对的答案部分）
                decode_input = self.encode_sentence(title, self.s2_max_length, True, False)  # 输出序列
                # decoder输出编码，加结束符号，decode_output和decode_input形成错位，中间内容都是title内容的编码
                decode_output = self.encode_sentence(title, self.s2_max_length, False,
                                                     True)  # 不进入模型，用于计算loss
                # input_seq + decode_input拼接构成输入问答对，decode_output是输出回答
                input = input_seq + decode_input
                self.data.append([torch.LongTensor(input),
                                  torch.LongTensor(decode_output)])
        return


    def encode_sentence(self, text, max_length, with_cls_token=True, with_sep_token=True, padding=True):
        input_ids = []
        if with_cls_token:
            input_ids.append(self.vocab["[CLS]"])
        for char in text:
            input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if with_sep_token:
            input_ids.append(self.vocab["[SEP]"])
        input_ids = self.padding(input_ids, max_length)
        return input_ids

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, max_length, pad_token=0):
        input_id = input_id[:max_length]
        input_id += [pad_token] * (max_length - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, sample_length, batch_size, s1_max_length, s2_max_length,  shuffle=True):
    dg = DataGenerator(data_path, sample_length, s1_max_length, s2_max_length)
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
# def load_corpus(path):
#     corpus = ""
#     with open(path, encoding="gbk") as f:
#         for line in f:
#             corpus += line.strip()
#     return corpus


#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

#文本生成测试代码
# def generate_sentence(openings, model, vocab, window_size):
#     reverse_vocab = dict((y, x) for x, y in vocab.items())
#     model.eval()
#     with torch.no_grad():
#         pred_char = ""
#         #生成了换行符，或生成文本超过30字则终止迭代
#         while pred_char != "\n" and len(openings) <= 30:
#             openings += pred_char
#             x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
#             x = torch.LongTensor([x])   # (1， seq_len)
#             if torch.cuda.is_available():
#                 x = x.cuda()
#             y = model(x)[0][-1]         # 最后一个字的(vocab_size,) 概率分布
#             index = sampling_strategy(y)
#             pred_char = reverse_vocab[index]
#     return openings

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


class Evaluator:
    def __init__(self, data_path, model, sample_length, batch_size, s1_max_length, s2_max_length, logger):
        self.path = data_path
        self.logger = logger
        self.valid_data = load_data(data_path, sample_length, batch_size,
                                    s1_max_length, s2_max_length, shuffle=False)
        self.reverse_vocab = dict([(y, x) for x, y in self.valid_data.dataset.vocab.items()])
        self.s1_max_length = s1_max_length
        self.s2_max_length = s2_max_length
        self.model = model

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        with torch.no_grad():
            for index, batch_data in enumerate(self.valid_data):
                if torch.cuda.is_available():
                    batch_data = [d.cuda() for d in batch_data]
                input_seqs, gold = batch_data
                for input_seq in input_seqs:   # input_seq.shape: (seq_len,)  eg. (150,)
                    # input shape: (1, seq_len) =>  output shape: (1, seq_len, vocab_size)
                    generate = self.model(input_seq.unsqueeze(0))[0]  # generate.shape: (seq_len, vocab_size)  eg. (150, 21128)
                    output_seq = [sampling_strategy(gen) for gen in generate]
                    logger.info("输入：%s", self.decode_seq(input_seq))
                    logger.info("输出：%s", self.decode_seq(output_seq[-self.s2_max_length:]))  # 预测输出只看最后s2，问答对的回答部分
                    break
        return

    def decode_seq(self, seq):
        return "".join([self.reverse_vocab[int(idx)] for idx in seq])


def train(data_path, save_weight=True):
    epoch_num = 50        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 5000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    s1_max_length = 105      # 输入内容（问题）文本长度
    s2_max_length = 20       # 输出摘要（答案）文本长度
    # vocab = build_vocab("vocab.txt")       #建立字表
    # corpus = load_corpus(corpus_path)     #加载语料
    #加载训练数据
    train_data = load_data(data_path, train_sample, batch_size,
                           s1_max_length, s2_max_length)
    vocab = train_data.dataset.vocab
    model = build_model(vocab, char_dim)    #建立模型
    # 加载效果测试类
    evaluator = Evaluator(data_path, model, train_sample, batch_size,
                          s1_max_length, s2_max_length, logger)
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
        evaluator.eval(epoch + 1)
        # logger.info(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        # logger.info(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
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
    train("sample_data.json", True)

