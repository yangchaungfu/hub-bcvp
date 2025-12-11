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


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()  # 对比损失，可以替换为其他损失函数

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    # 三元组损失（Triplet Loss）：输入为 anchor、positive 和 negative
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
        return torch.mean(diff[diff.gt(0)])  # greater than

    # 修改后的 forward 方法同时传入 3 个句子
    # sentence1 是 anchor，sentence2 是 positive，sentence3 是 negative
    def forward(self, sentence1, sentence2, sentence3):
        vector1 = self.sentence_encoder(sentence1)
        vector2 = self.sentence_encoder(sentence2)
        vector3 = self.sentence_encoder(sentence3)
        loss = self.cosine_triplet_loss(vector1, vector2, vector3)
        return loss



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

    # 示例数据：三个句子和它们的标签
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    s3 = torch.LongTensor([[1, 2, 3, 5], [4, 2, 3, 5]])
    l = torch.LongTensor([[1], [0]])  # 标签可以是1（相似）或-1（不相似）

    # 使用模型进行三元组损失计算
    y = model(s1, s2, s3)
    print(y)
    # print(model.state_dict())