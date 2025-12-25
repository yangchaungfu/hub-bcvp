#coding:utf8

import torch
import torch.nn as nn
import numpy as np
from config import Config
from loader import load_vocab, load_corpus
from torch.optim import Adam, SGD
from transformers import BertModel

"""
基于transformer的Bert语言模型
实现BERT语言模型训练，支持Mask Attention机制
"""

def subsequent_mask(size):
    """
    生成因果掩码(Causal Mask)，用于防止模型在预测时看到未来的token
    该掩码将当前token之后的所有位置标记为False（被遮蔽）
    
    参数:
        size: 序列长度
    返回:
        形状为(1, size, size)的布尔张量，上三角部分为False（被遮蔽）
    """
    attn_shape = (1, size, size)
    # 使用上三角矩阵，k=1表示从主对角线开始向上偏移一位
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 取反，使得被遮蔽的位置为True，未被遮蔽的位置为False
    return torch.from_numpy(subsequent_mask) == 0

def pad_mask(inputs, pad_id=0):
    """
    生成填充掩码(Padding Mask)，用于忽略输入序列中的padding token
    
    参数:
        inputs: 输入的张量，形状为(batch_size, seq_len)
        pad_id: padding token的索引，默认为0
    返回:
        形状为(1, 1, seq_len)的布尔张量，padding位置为False
    """
    input_mask = (inputs != pad_id).unsqueeze(1).unsqueeze(2)
    return input_mask

class BertLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super(BertLanguageModel, self).__init__()
        self.pad_id = 0
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.corpus = load_corpus(config["corpus_path"])
        self.classify = nn.Linear(config["hidden_size"], len(self.vocab))

        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        """
        前向传播函数，实现BERT模型和Mask Attention机制
        
        参数:
            x: 输入序列，形状为(batch_size, seq_len)
            y: 目标序列，用于计算loss，形状为(batch_size, seq_len)
        返回:
            如果y存在，返回loss值；否则返回预测的概率分布
        """
        # 1. 生成填充掩码，标记非padding的位置
        # 形状: (batch_size, 1, seq_len)
        input_mask = (x != self.pad_id).unsqueeze(1)
        
        # 2. 获取序列长度，生成因果掩码
        # 因果掩码确保位置i只能关注位置i及之前的位置
        seq_len = x.size(1)
        causal_mask = subsequent_mask(seq_len).to(x.device)
        
        # 3. 组合掩码：同时考虑padding和因果关系
        # 只有同时满足非padding且不是未来位置的位置才能被关注
        combined_mask = input_mask & causal_mask
        
        # 4. 移除多余的维度，得到BERT所需的2D掩码
        # BERT期望的attention_mask形状: (batch_size, seq_len)
        attention_mask = combined_mask.squeeze(1)
        
        # 5. 将掩码传入BERT模型
        # BERT会自动根据attention_mask忽略被遮蔽的位置
        x, _ = self.bert(x, attention_mask=attention_mask)
        
        # 6. 分类层：将BERT输出映射到词表大小
        y_pred = self.classify(x)
        
        # 7. 计算损失或返回预测概率
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)