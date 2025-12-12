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
        # 保留原损失函数，同时添加triplet loss支持
        self.loss = nn.CosineEmbeddingLoss()
        # 添加margin参数
        self.margin = config.get("margin", 0.1)

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        # 计算a和p的余弦距离
        ap = self.cosine_distance(a, p)
        # 计算a和n的余弦距离
        an = self.cosine_distance(a, n)
        # 如果没有设置margin，则使用默认margin
        if margin is None:
            diff = ap - an + self.margin
        # 如果设置了margin，则使用传入的margin
        else:
            diff = ap - an + margin.squeeze()
        # 返回diff中大于0的部分的平均值
        return torch.mean(diff[diff.gt(0)])  # greater than

    # 新增：三元组前向传播
    def forward_triplet(self, anchor, positive, negative):
        """
        三元组训练模式
        anchor: 锚样本 (batch_size, max_length)
        positive: 正样本 (batch_size, max_length)
        negative: 负样本 (batch_size, max_length)
        返回: triplet loss
        """
        anchor_vec = self.sentence_encoder(anchor)
        positive_vec = self.sentence_encoder(positive)
        negative_vec = self.sentence_encoder(negative)

        return self.cosine_triplet_loss(anchor_vec, positive_vec, negative_vec)

    # 新增：批量三元组前向传播（高效版本）
    def forward_batch_triplet(self, sentences):
        """
        批量处理模式：输入一个批次的所有句子，返回三元组损失
        输入: (batch_size*3, max_length)  # 按照[anchor1, positive1, negative1, anchor2, positive2, negative2, ...]排列
        返回: triplet loss
        """
        batch_size = sentences.shape[0] // 3
        all_vecs = self.sentence_encoder(sentences)  # (batch_size*3, hidden_size)

        total_loss = 0
        for i in range(batch_size):
            anchor = all_vecs[i * 3]
            positive = all_vecs[i * 3 + 1]
            negative = all_vecs[i * 3 + 2]
            total_loss += self.cosine_triplet_loss(
                anchor.unsqueeze(0),
                positive.unsqueeze(0),
                negative.unsqueeze(0)
            )

        return total_loss / batch_size

    # sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, sentence3=None, target=None, mode="pair"):
        """
        增强的前向传播，支持多种模式
        mode:
          - "pair": 原双句子模式（默认）
          - "triplet": 三元组模式（需要sentence3）
          - "batch_triplet": 批量三元组模式（sentence1包含所有样本）
        """
        # 三元组模式
        if mode == "triplet" and sentence3 is not None:
            return self.forward_triplet(sentence1, sentence2, sentence3)

        # 批量三元组模式
        elif mode == "batch_triplet":
            return self.forward_batch_triplet(sentence1)

        # 原双句子模式（保持完全兼容）
        elif sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            #如果有标签，则计算loss
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            #如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        #单独传入一个句子时，认为正在使用向量化能力
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

    # 添加margin配置
    Config["margin"] = 0.1
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)

    # 测试原功能
    print("=== 测试原双句子功能 ===")
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print(f"原损失: {y}")

    # 测试三元组功能
    print("\n=== 测试三元组功能 ===")
    anchor = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    positive = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    negative = torch.LongTensor([[4, 5, 6, 7], [5, 5, 5, 5]])

    # 方法1：使用triplet模式
    loss1 = model(anchor, positive, negative, mode="triplet")
    print(f"三元组损失(模式1): {loss1}")

    # 方法2：使用forward_triplet方法
    loss2 = model.forward_triplet(anchor, positive, negative)
    print(f"三元组损失(模式2): {loss2}")

    # 测试批量三元组功能
    print("\n=== 测试批量三元组功能 ===")
    # 准备批量数据：[anchor1, positive1, negative1, anchor2, positive2, negative2]
    batch_triplets = torch.cat([anchor, positive, negative], dim=0)
    print(f"批量数据形状: {batch_triplets.shape}")  # 应该是 (6, 4)

    loss3 = model(batch_triplets, mode="batch_triplet")
    print(f"批量三元组损失: {loss3}")

    # 测试向量化功能（保持不变）
    print("\n=== 测试向量化功能 ===")
    vector = model(s1)
    print(f"向量形状: {vector.shape}")