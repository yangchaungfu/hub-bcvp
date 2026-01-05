# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
from transformers import BertModel


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        model_type = config["model_type"]
        out_channels = config["out_channels"]
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        self.config = config
        self.emb_layer = nn.Embedding(vocab_size, hidden_size)
        if model_type == "bert":
            self.encoder = Bert(config)
            bert = BertModel.from_pretrained(config["pretrain_model_path"])
            hidden_size = bert.config.hidden_size
        elif model_type == "LSTM":
            self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif model_type == "TextCNN":
            self.encoder = TextCNN(config)
            hidden_size = out_channels
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        if self.config["model_type"] == "bert":
            x = self.encoder(x)
        else:
            x = self.emb_layer(x)
            x = self.encoder(x)
        # LSTM会返回元组
        if isinstance(x, tuple):
            x = x[0]
        pool = nn.AvgPool1d(x.shape[1])
        x = pool(x.transpose(1, 2)).squeeze(2)
        return x


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        hidden_size = config["hidden_size"]
        padding = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, hidden_size),
                             padding=(padding, 0))

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        return x.squeeze(-1).transpose(1, 2)


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=True)

    def forward(self, x):
        outputs = self.bert(input_ids=x['input_ids'].squeeze(1),
                            attention_mask=x['attention_mask'].squeeze(1),
                            token_type_ids=x['token_type_ids'].squeeze(1))
        return outputs.last_hidden_state


# 计算两个向量的cosine距离（1-cosine）
def cosine_distance(x, y):
    x_norm = nn.functional.normalize(x, dim=-1)
    y_norm = nn.functional.normalize(y, dim=-1)
    cosine = torch.sum(x_norm * y_norm)
    return 1 - cosine


class MatchModel(nn.Module):
    def __init__(self, config):
        super(MatchModel, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, x, y=None, z=None):
        x = self.encoder(x)
        # 如果只传入一个参数，则只需要将文本向量化
        if y is None:
            return x
        # 如果只传入x,y，则计算两个向量的cosine距离
        if z is None:
            y = self.encoder(y)
            return cosine_distance(x, y)
        # 如果y和z都不为空，则是在进行训练，可能是双塔模型训练，也可能是三元组的Triplet loss
        # 如果是双塔模型，需要判断是直接使用cosine计算loss，还是MLP连接层进行二分类
        # 如果是三元组，则直接计算Triplet loss
        y = self.encoder(y)
        if self.config["train_type"] == "Siam":
            if self.config["matching_type"] == "cosine":
                # z是label，计算loss需要将（batch,1）转化为（batch,）的向量
                return self.loss(x, y, z.squeeze())
            else:
                # 拼接方式：0:(u,v)   1:(u,v,|u-v|)   2:(u,v,u*v)
                if self.config["concat_type"] == 0:
                    res = torch.cat([x, y], dim=-1)
                elif self.config["concat_type"] == 1:
                    res = torch.cat([x, y, torch.abs(x - y)], dim=-1)
                else:
                    res = torch.cat([x, y, x * y], dim=-1)
                input_size = res.shape[-1]
                linear = nn.Linear(input_size, 2)
                res = linear(res)
                loss = nn.CrossEntropyLoss()
                # 此时的z为label，里面的值都是-1和1，进行二分类时需要转为0和1
                z = torch.where(z == -1, torch.tensor(0), z)
                return loss(res, z.squeeze())
        else:
            z = self.encoder(z)
            # 为了方便理解，将xyz转为三元组形式
            a, p, n = x, y, z
            d_ap = cosine_distance(a, p)
            d_an = cosine_distance(a, n)
            margin = self.config["margin"]
            # loss = max(d_ap - d_an + margin, 0)
            diff = torch.relu(d_ap - d_an + margin)
            # diff是一个（batch，1）的张量，后面的1维是cosine距离，需要取出大于0的部分的平均值作为loss
            loss = diff.mean()
            return loss


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["hidden_size"] = 5
    Config["max_length"] = 4
    model = MatchModel(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [-1]])
    y = model(s1, s2, l)
    print(y.item())
