# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        if model_type == "fast_text":
            self.encoder = lambda x: x  # 编码层不做处理，全靠后面的Pooling
        elif model_type == "lstm":
            # 修改为双向LSTM
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                                   bidirectional=True)
            hidden_size = hidden_size * 2  # 双向维度翻倍
        elif model_type == "cnn":
            self.encoder = CNN(config)

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.embedding(x)  # input shape:(batch_size, sen_len)
        x = self.encoder(x)  # shape:(batch_size, sen_len, hidden_size)

        if isinstance(x, tuple):  # RNN类的模型会返回(output, hidden)，我们只要output
            x = x[0]

        # 池化层
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])

        x = x.transpose(1, 2)
        x = self.pooling_layer(x).squeeze()  # shape:(batch_size, hidden_size)

        predict = self.classify(x)  # shape:(batch_size, class_num)

        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
