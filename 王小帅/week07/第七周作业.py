# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import sklearn.model_selection as ms
from torch.utils.data import DataLoader, TensorDataset
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self,vocab_size=21128, hidden_size=256, class_num=2, model_type = 'bert', num_layers=2, kernel_size=3):
        super(TorchModel, self).__init__()
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.model_type = model_type
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.usebert = False
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(self.hidden_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(self.hidden_size, self.kernel_size)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(self.hidden_size, self.kernel_size)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(self.hidden_size, self.num_layers)
        elif model_type == "rcnn":
            self.encoder = RCNN(self.hidden_size, self.kernel_size)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(r"/D:\Program_Files\nlp25\week6\bert-base-chinese", return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(r"D:\Program_Files\nlp25\week6\bert-base-chinese")
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(r"D:\Program_Files\nlp25\week6\bert-base-chinese")
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(r"D:\Program_Files\nlp25\week6\bert-base-chinese")
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = "max"
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            #sequence_output:batch_size, max_len, hidden_size
            #pooler_output:batch_size, hidden_size
            x = self.encoder(x)
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim)

        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        #可以采用pooling的方式得到句向量
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze() #input shape:(batch_size, sen_len, input_dim)

        #也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]
        predict = self.classify(x)   #input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        kernel_size = kernel_size
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(hidden_size, kernel_size)
        self.gate = CNN(hidden_size, kernel_size)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(StackGatedCNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(num_layers) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        #仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)               #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1) #二层线性
            x = self.bn_after_ff[i](x + l2)        #残差后过bn
        return x


class RCNN(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super(RCNN, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(hidden_size, kernel_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x

class BertLSTM(nn.Module):
    def __init__(self, pretrain_model_path):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self, pretrain_model_path):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.hidden_size = self.bert.config.hidden_size
        self.kernel_size = self.bert.config.kernel_size
        self.cnn = CNN(self.hidden_size, self.kernel_size)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x

class BertMidLayer(nn.Module):
    def __init__(self, pretrain_model_path):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]#(13, batch, len, hidden)
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


def train(data, model, optimizer):
    model.train()
    train_loss = []
    for index, (input_ids, labels) in enumerate(data):
        optimizer.zero_grad()
        loss = model(input_ids, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if index % 10 == 0:
            print(f'Update Batch: {index} [{index * len(input_ids)}/{len(data)} ({100. * index / len(data):.0f}%)]\tLoss: {loss.item():.6f}')
            # 计算每个训练周期的平均损失，并将其添加到epoch_loss中
        (train_loss.append(loss.item()))
    return train_loss


def calculateAccuracy(data, model):
    model.eval()
    pre = []
    truth = []
    with torch.no_grad():
        for index, (input_ids, labels) in enumerate(data):
            prediction = model(input_ids).argmax(dim=-1)
            pre.append(prediction.numpy())
            truth.append(labels.numpy().squeeze())
        acc = accuracy_score(np.asarray(truth), np.asarray(pre))
    return acc

if __name__ == '__main__':
    path = r'D:\Program_Files\nlp25\week7\文本分类练习.csv'
    data = pd.read_csv(path, encoding='utf-8')
    model_path = r'D:\Program_Files\nlp25\week6\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path, return_dict=False)
    label = data['label'].values
    print(f'标签为{label}, {label.shape}')
    sen_len = sorted([len(tokenizer.encode(i)) for i in data['review']], reverse=True)
    max_len = max(sen_len)
    data = [tokenizer.encode(i, max_length=max_len, padding='max_length') for i in data['review']]
    data = np.stack(data, axis=0)  # [batch, 458]

    x_train, y_train = ms.train_test_split(data, label, test_size=0.4, stratify=label)
    print(f'训练集大小为：{x_train.shape},{y_train.shape}, 正样本数有{sum(y_train)}个')
    model_type = 'bert'
    lr = 0.01
    batch = 1
    hidden_size = 256
    model = TorchModel(hidden_size=hidden_size, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    train_loss_epoch = []
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train.astype(np.long)), torch.from_numpy(y_train.astype(np.long).reshape(-1, 1))),
                              batch_size=batch, shuffle=True, drop_last=True)
    for epoch in range(2):
        loss = train(train_loader, model, optimizer)
        train_loss.extend(loss)
        train_loss_epoch.append(sum(loss) / len(loss))
        print(f'epoch {epoch + 1:3d}, Average loss {sum(loss) / len(loss):.3f}')
    acc = calculateAccuracy(train_loader, model)
    print(f'使用的模型是：{model_type}，准确率：{acc}')
