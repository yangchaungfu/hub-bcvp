# -*- coding:utf-8 -*-

"""
model的任务：选择模型、构建网络层及网络层的运算逻辑
"""

import torch
import torch.nn as nn
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]
        bidirectional = config["bidirectional"]
        class_num = config["class_num"]
        vocab_size = config["vocab_size"]
        self.pooling_type = config["pooling_type"]
        self.isBert = False
        if model_type not in ["bert", "bertRNN"]:
            # bert模型有自己的Embedding
            self.emb_layer = nn.Embedding(vocab_size, hidden_size)
        if model_type == "fastText":
            # 这里将fastText直接当成简单的映射分类模型,所以在这一层什么都不用做
            self.encoder = lambda x: x
        if model_type == "RNN":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
            # 如果是双向的，最后hidden_state会进行拼接
            if bidirectional:
                hidden_size *= 2
        if model_type == "CNN":
            self.encoder = CNN(config)
            # CNN最后的输出维度为通道数
            hidden_size = config["out_channels"]
        if model_type == "TextCNN":
            self.encoder = TextCNN(config)
        if model_type == "RCNN":
            self.encoder = RCNN(config)
            hidden_size = config["out_channels"]
        if model_type == "LSTM":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
            if bidirectional:
                hidden_size *= 2
        if model_type == "bert":
            self.isBert = True
            self.encoder = Bert(config)
            bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            # 如果是bert，后续的hidden_size改为bert的768
            hidden_size = bert.config.hidden_size
        if model_type == "bertRNN":
            self.encoder = BertRNN(config)
            if bidirectional:
                hidden_size *= 2
        # 分类层
        self.classifier = nn.Linear(hidden_size, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        if not self.isBert:
            x = self.emb_layer(x)
        x = self.encoder(x)
        # 判断x是否像RNN一样返回的元组
        if isinstance(x, tuple):
            x = x[0]
        # 池化层，以多个序列进行纵向池化，将（batch_size, seq_len, char_dim） -> (batch_size, char_dim)
        if self.pooling_type == "avg":
            self.pooler_layer = nn.AvgPool1d(x.shape[1])
        elif self.pooling_type == "max":
            self.pooler_layer = nn.MaxPool1d(x.shape[1])
        # 池化层默认都是池化最后一个维度，所以需要转置，最后将池化后的1维去掉
        x = self.pooler_layer(x.transpose(1, 2)).squeeze(2)
        y_pred = self.classifier(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze(1))
        return torch.softmax(y_pred, dim=1)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        # 为了让输出维度和输入序列的维度相同，需要在首尾进行补零
        padding = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # 1D的CNN需要将输入的后面两个维度进行转置，
        # （batch_size, seq_len, feature_dim） --> (batch_size, feature_dim, seq_len)
        # 最终输出为（batch_size, out_channels, seq_len） --> （batch_size, seq_len, out_channels）
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        hidden_size = config["hidden_size"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        padding = int((self.kernel_size - 1) / 2)
        # TextCNN使用的是2D的CNN，in_channels固定为1，将feature_dim放到kernel_size里面
        self.text_cnn = nn.Conv2d(1, out_channels, kernel_size=(kernel_size, hidden_size), padding=padding)

    def forward(self, x):
        # 2D的CNN需要在输入的第二个维度添加一个1维
        # （batch_size, seq_len, feature_dim） --> （batch_size, 1, seq_len, feature_dim）
        # 最终输出为（batch_size, out_channels, seq_len, 1） --> （batch_size, seq_len, out_channels）
        return self.text_cnn(x.unsqueeze(1)).squeeze(3).transpose(1, 2)


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        bidirectional = config["bidirectional"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        padding = (self.kernel_size - 1) / 2
        # TextCNN使用的是2D的CNN，in_channels固定为1，将feature_dim放到kernel_size里面
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            config["hidden_size"] = hidden_size * 2
        self.cnn = CNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.num_layers = config["num_layers"]
        # 需要返回transformer不同层数的隐藏状态
        # 如果return_dict为False，self.bert(x)将返回元组类型，就必须将self.bert(x).hidden_states改为self.bert(x)[2]
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], output_hidden_states=True,
                                              return_dict=True)

    def forward(self, x):
        """
        self.bert(x)返回三个主要数据，分别last_hidden_state、pooler_output、hidden_states
        hidden_states是元组类型，包含1个嵌入层的输出和12个transformer层的隐藏状态
        hidden_states形状为（13，batch_size, max_len, hidden_size）
        """
        outputs = self.bert(input_ids=x['input_ids'].squeeze(1),
                            attention_mask=x['attention_mask'].squeeze(1),
                            token_type_ids=x['token_type_ids'].squeeze(1))
        hidden_states = outputs.hidden_states
        # 获取指定层数的隐藏状态，然后取出cls
        batch_token = hidden_states[self.num_layers]
        return batch_token


class BertRNN(nn.Module):
    def __init__(self, config):
        super(BertRNN, self).__init__()
        self.num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]
        bidirectional = config["bidirectional"]
        # 需要返回transformer不同层数的隐藏状态
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.RNN(self.bert.config.hidden_size, hidden_size, num_layers=self.num_layers,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        last_hidden_states = self.bert(x).last_hidden_state
        return self.rnn(last_hidden_states)


if __name__ == "__main__":
    import torch

    model = BertModel.from_pretrained("./data/bert-base-chinese", return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    # print(model(x).last_hidden_state)

    print(model.config)
