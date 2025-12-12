# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


"""
孪生网络
"""
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1) # normalize函数将张量沿着最后一个维度(dim=-1)进行L2归一化，归一化后每个向量的模长都为1
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        # torch.sum(..., axis=-1) 沿着最后一个维度求和，结果: a1*b1 + a2*b2 + a3*b3
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)  # 内积（点击）的定义是对应元素相乘后求和，sum([a1*b1, a2*b2, a3*b3])
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        # 计算a和p的余弦距离
        ap = self.cosine_distance(a, p)
        # 计算a和n的余弦距离
        an = self.cosine_distance(a, n)
        # 如果没有设置margin，则设置diff为ap - an + 0.1
        if margin is None:
            diff = ap - an + 0.1
        # 如果设置了margin，则设置diff为ap - an + margin.squeeze()
        else:
            diff = ap - an + margin.squeeze()
        # 返回diff中大于0的部分的平均值
        return torch.mean(diff[diff.gt(0)]) #greater than

    """
    这个方法有2个功能：
    1. 输入sentence1，将sentence1文本转化为向量
        => 评价model时，测试输入问题转向量，test_question_vectors = self.model(input_id)
        => 评价model时，知识库所有问题转向量，self.knwb_vectors = self.model(question_matrixs)
    2. 输入sentence1，sentence2和sentence3，将sentence1、sentence2和sentence3文本转化为向量，并计算cosine_triplet_loss损失  
        => model训练时，loss = model(input_id1, input_id2, input_id3)
        (相似样本1，相似样本2，不相似样本3)
    """
    #sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, sentence3=None):  # target为-1或1
        # 同时传入三个句子
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            vector3 = self.sentence_encoder(sentence3)
            # 计算loss
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())