# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertConfig

"""
基于pytorch的BERT语言模型，使用自回归训练（下一个词预测）
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.vocab_size = len(vocab)

        # 使用BERT的配置，设置为解码器模式
        config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=input_dim,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=input_dim * 4,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            is_decoder=True,  # 设置为解码器，启用因果掩码
            add_cross_attention=False,
        )
        self.bert = BertModel(config)

        # 添加一个线性层将BERT输出映射到词汇表
        self.lm_head = nn.Linear(input_dim, self.vocab_size, bias=False)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 创建注意力掩码（全1，因为输入都是有效字符）
        attention_mask = torch.ones(x.shape[:2], device=x.device)

        # BERT前向传播
        # 注意：这里我们使用input_ids而不是inputs_embeds
        bert_output = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 获取最后一个隐藏层状态
        hidden_states = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 预测下一个词
        logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]

        if y is not None:
            # 计算损失：预测下一个词
            # 将logits reshape为 [batch_size*seq_len, vocab_size]
            # 将y reshape为 [batch_size*seq_len]
            loss = self.loss(logits.view(-1, self.vocab_size), y.view(-1))
            return loss
        else:
            # 返回下一个词的概率分布
            return torch.softmax(logits, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    # 不需要添加[MASK] token，因为我们不使用MLM
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # 只取最后window_size个字符作为输入
            context = openings[-window_size:] if len(openings) > window_size else openings
            x = [vocab.get(char, vocab["<UNK>"]) for char in context]
            # 如果长度不够，用<pad>填充
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]  # 取最后一个位置的预测
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
        # 添加微小值避免概率和为0
        prob_distribution = np.maximum(prob_distribution, 1e-10)
        prob_distribution = prob_distribution / prob_distribution.sum()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            # 填充到窗口大小
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index].item()
            # 避免log(0)
            if target_prob < 1e-10:
                target_prob = 1e-10
            prob += math.log(target_prob)
    # 使用自然对数计算困惑度
    if len(sentence) > 1:
        return math.exp(-prob / (len(sentence) - 1))
    else:
        return float('inf')


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab = build_vocab("vocab.txt")
    # 加载语料
    corpus = load_corpus(corpus_path)

    # 建立模型
    model = build_model(vocab, char_dim)

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    print(f"词汇表大小: {len(vocab)}")
    print(f"语料长度: {len(corpus)}")
    print("文本词表模型加载完毕，开始训练")

    # 训练前先测试生成
    print("初始生成测试:")
    print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()

            # 添加梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss: {avg_loss:.6f}")

        # 每个epoch后测试生成
        model.eval()
        print("生成测试1:", generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print("生成测试2:", generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
