
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, max_len, hidden_size)
        x = self.layer(x)      # 线性变换
        x = self.dropout(x)
        # 最大池化获取句子向量
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.cosine_loss = nn.CosineEmbeddingLoss()  # 保留余弦损失用于对比

    def cosine_distance(self, tensor1, tensor2):
        """计算余弦距离 (1 - 余弦相似度)"""
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def triplet_loss(self, anchor, positive, negative, margin=0.5):
        """三元组损失: 使anchor与positive的距离小于anchor与negative的距离"""
        ap_distance = self.cosine_distance(anchor, positive)
        an_distance = self.cosine_distance(anchor, negative)
        # 损失 = max(0, ap距离 - an距离 + margin)
        loss = torch.mean(torch.max(torch.zeros_like(ap_distance), ap_distance - an_distance + margin))
        return loss

    def forward(self, anchor, positive=None, negative=None, target=None):
        # 三元组训练模式
        if positive is not None and negative is not None:
            a = self.sentence_encoder(anchor)
            p = self.sentence_encoder(positive)
            n = self.sentence_encoder(negative)
            return self.triplet_loss(a, p, n)
        
        # 成对比较模式（用于评估和预测）
        if positive is not None:
            vector1 = self.sentence_encoder(anchor)
            vector2 = self.sentence_encoder(positive)
            if target is not None:
                return self.cosine_loss(vector1, vector2, target.squeeze())
            else:
                return self.cosine_distance(vector1, vector2)
        
        # 单独编码句子（用于生成向量）
        return self.sentence_encoder(anchor)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 1000
    model = SiameseNetwork(Config)
    # 测试三元组输入
    a = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    p = torch.LongTensor([[1,2,3,4], [2,2,3,0]])
    n = torch.LongTensor([[5,6,7,8], [9,8,7,0]])
    loss = model(a, p, n)
    print("Triplet loss:", loss.item())
