import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, T5Model, BertTokenizer
from loadey import DataGenerator, load_data

"""
使用 bert 进行 sft 训练
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese", return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            mask = []
            for i in range(x.shape[0]):
                ind = torch.nonzero(x[i]==102).squeeze().item()
                mask1 = torch.cat([torch.ones(ind + 1, ind + 1), torch.zeros(ind + 1, x.shape[1]- ind - 1)], dim=-1)
                mask2 = torch.cat([torch.ones(x.shape[1]- ind - 1, ind + 1), torch.tril(torch.ones(x.shape[1]- ind - 1, x.shape[1]- ind - 1))], dim=1)
                mask.append(torch.cat([mask1, mask2], dim=0))
            mask = torch.stack(mask, dim=0)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask) #output shape:(batch_size, sen_len, hidden_size)
            y_pred = self.classify(x) #output shape:(batch_size, sen_len, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)  # output shape:(batch_size, sen_len, hidden_size)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus):
    tokenizer = BertTokenizer.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese")
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   #将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(hidden_size, vocab_size):
    model = LanguageModel(hidden_size, vocab_size)
    return model

#文本生成测试代码
def generate_sentence(openings, model):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese")
    with torch.no_grad():
        out_char = ""
        pre_cha = ""
        #生成了换行符，或生成文本超过150字则终止迭代
        while out_char != "[SEP]" and len(pre_cha) <= 150:
            x = tokenizer.encode(openings, add_special_tokens=False) + [102] + tokenizer.encode(pre_cha, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            out_char = ''.join(tokenizer.decode(index))
            pre_cha +=  out_char
    return pre_cha

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
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(data, save_weight=True):
    epoch_num = 500        #训练轮数
    batch_size = 32       #每次训练样本个数
    hidden_size = 768        #每个字的维度
    vocab_size = 21128
    # vocab = build_vocab("vocab.txt")       #建立字表
    model = build_model(hidden_size, vocab_size)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch, (x, y) in enumerate(data):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("美国最适合创业的十大行业", model))
        print(generate_sentence("北京目前六成人口属社会中下层 2025年则一半将是中产？", model))
    if not save_weight:
        return
    else:
        torch.save(model.state_dict(), f'model.pth')
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    data = load_data(r'sample_data.json', logger=1)
    train(data)