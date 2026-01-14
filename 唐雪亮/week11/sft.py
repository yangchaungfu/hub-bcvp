# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertConfig, BertTokenizer

"""
基于pytorch的Bert语言模型
生成任务
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        model_path = r"D:\bert-base-chinese"
        self.bert_config = BertConfig.from_pretrained(model_path)
        self.bert_config.num_hidden_layers = 6  # 设置Bert层数
        self.Bert = BertModel(config=self.bert_config)  # 实例化模型
        self.classify = nn.Linear(self.bert_config.hidden_size, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, y=None):
        # mask = torch.LongTensor(torch.tril(torch.ones(x.shape[0], x.shape[1], x.shape[1])))
        # mask_final = torch.matmul(attention_mask, mask)
        output = self.Bert(x, attention_mask=attention_mask)  # output shape:(batch_size, sen_len, hidden_size)
        x_last = self.dropout(output.last_hidden_state)  # 随机熄灭神经元，按比例放大其余神经元
        y_pred = self.classify(x_last)  # output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=-100)
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):  # 将语料去除空格输出
    corpus = ""
    with open(path, encoding="GBK") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus, tokenizer):
    start = random.randint(0, len(corpus) - 1 - window_size)  # 随机窗口起始位置
    end = start + window_size  # 窗口结束位置
    window = corpus[start:end]  # 确定窗口中的实际字符串
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    # 编码输入和目标
    x = tokenizer.encode_plus(window,
                              max_length=window_size + 1,
                              padding="max_length",
                              truncation=True,
                              return_attention_mask=True,
                              add_special_tokens=False)

    y = tokenizer.encode_plus(target,
                              max_length=window_size + 1,
                              padding="max_length",
                              truncation=True,
                              return_attention_mask=True,
                              add_special_tokens=False)

    x_input_ids = x["input_ids"]
    x_attention_mask = x["attention_mask"]
    y_input_ids = y["input_ids"]

    # 设置标签，只有答案部分有效
    labels = [-100] * (window_size // 2) + y_input_ids[window_size // 2:]

    return x_input_ids, x_attention_mask, labels


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus, tokenizer):
    dataset_x_input_ids = []
    dataset_x_attention_mask = []
    dataset_y_labels = []
    for i in range(sample_length):
        x_input_id, x_att_mask, y_label \
            = build_sample(vocab, window_size, corpus, tokenizer)
        dataset_x_input_ids.append(x_input_id)
        dataset_x_attention_mask.append(x_att_mask)
        dataset_y_labels.append(y_label)
    return (
        torch.LongTensor(dataset_x_input_ids),
        torch.LongTensor(dataset_x_attention_mask),
        torch.LongTensor(dataset_y_labels)
    )


# 建立模型
def build_model(vocab):
    model = LanguageModel(vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode_plus(openings,
                                      max_length=window_size,
                                      padding=False,
                                      truncation=True,
                                      return_attention_mask=True,
                                      add_special_tokens=False)
            x_input_ids = x["input_ids"]
            x_in = torch.LongTensor([x_input_ids])
            if torch.cuda.is_available():
                x_in = x_in.cuda()
            # print(x_in.shape)
            y = model(x_in)[0][-1]  # 取出映射的字表维度的向量
            index = sampling_strategy(y)  # 利用贪心算法得出概率最大的那个字
            pred_char = tokenizer.decode([index])[0]
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


# 计算文本ppl
def calc_perplexity(sentence, model, tokenizer, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = tokenizer.encode_plus(window,
                                      max_length=window_size,
                                      padding="max_length",
                                      truncation=True,
                                      return_attention_mask=True,
                                      add_special_tokens=False)
            x_input_ids = x["input_ids"]
            x_in = torch.LongTensor([x_input_ids])
            target = sentence[i]
            target_index = tokenizer.convert_tokens_to_ids(target)
            if torch.cuda.is_available():
                x_in = x_in.cuda()
            pred_prob_distribute = model(x_in)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=False):
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    window_size = 32  # 样本文本长度
    vocab = build_vocab("vocab_bert.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    tokenizer = (BertTokenizer.from_pretrained
                 (r"D:\bert-base-chinese", do_lower_case=True))
    model = build_model(vocab)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, x_att, y = build_dataset(batch_size, vocab, window_size, corpus, tokenizer)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, x_att, y = x.cuda(), x_att.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, x_att, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("李清脚步一顿，似乎是明白了", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if save_weight:
        model_path = os.path.join(r"E:\AI_NLP课程\第十一周 大模型相关第一讲\sft\models",
                                  "DIY_nnlm_bert.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存至 {model_path}")
        return
    else:
        return


if __name__ == "__main__":
    train("corpus.txt", True)
