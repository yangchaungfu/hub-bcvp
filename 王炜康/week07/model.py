import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from torch.utils.data import DataLoader, TensorDataset


# 构建网络模型

class MyModel(nn.Module):
    def __init__(self,vocab_size=21128, hidden_size=256, class_num=2, model_type = 'bert', num_layers=2):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.model_type = model_type
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.usebert = False
        self.embedding = nn.Embedding(self.vocab_size, hidden_size, padding_idx=0)
        if model_type == 'fast_text':
            self.encoder = lambda x:x
        elif model_type == 'lstm':
            self.encoder = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        elif model_type == 'gru':
            self.encoder = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        elif model_type == 'rnn':
            self.encoder = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        elif model_type == 'cnn':
            self.encoder = CNN(self.hidden_size)
        elif model_type == 'gated_cnn':
            self.encoder = GatedCnn(self.hidden_size)
        elif model_type == 'StackGatedCNN':
            self.encoder = StackGatedCNN(self.hidden_size, self.num_layers)
        elif model_type == 'RCNN':
            self.encoder = RCNN(self.hidden_size)
        elif model_type == 'bert':
            self.usebert = True
            self.encoder = BertModel.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese", return_dict=False)
            self.hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.usebert = True
            self.encoder = BertLSTM()
            self.hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.usebert = True
            self.encoder = BertCNN()
            self.hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.usebert = True
            self.encoder = BertMidLayer()
            self.hidden_size = self.encoder.bert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.class_num)
        self.pooling_style = 'avg'
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        if self.usebert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.pooling_style == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        x = self.pooling(x.transpose(1, 2)).squeeze()

        predict = self.classifier(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict

class CNN(nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = 3
        pad = int((self.kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(self.hidden_size, self.hidden_size, self.kernel_size, bias=False, padding=pad)
    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)
class GatedCnn(nn.Module):
    def __init__(self, hidden_size):
        super(GatedCnn, self).__init__()
        self.hidden_size = hidden_size
        self.cnn = CNN(self.hidden_size)
        self.gate = CNN(self.hidden_size)
    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(StackGatedCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gcnn_layers = nn.ModuleList(
            [GatedCnn(self.hidden_size) for _ in range(num_layers)]
        )
        self.ff_liner_layers1 = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(num_layers)]
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x
            x = self.bn_after_gcnn[i](x)
            l1 = self.ff_liner_layers1[i](x)
            l1 = torch.relu(l1)
            l2 = self.ff_liner_layers2[i](x)
            x = self.bn_after_ff[i](x + l2)


class RCNN(nn.Module):
    def __init__(self, hidden_size):
        super(RCNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size, batch_first=True)
        self.cnn = GatedCnn(self.hidden_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):
    def __init__(self):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese", return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x


class BertCNN(nn.Module):
    def __init__(self):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese", return_dict=False)
        self.hidden_size = self.bert.config.hidden_size
        self.cnn = CNN(self.hidden_size)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese", return_dict=False)
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


def test(data, model):
    model.eval()
    pre = []
    truth = []
    with torch.no_grad():
        start = time.time()
        for index, (input_ids, labels) in enumerate(data):
            prediction = model(input_ids).argmax(dim=-1)
            pre.append(prediction.numpy())
            truth.append(labels.numpy().squeeze())
            if index == 99:
                end = time.time()
        acc = sklearn.metrics.accuracy_score(np.asarray(truth), np.asarray(pre))
        print(f'100条测试集时间:{end - start}')
    return acc, end - start







if __name__ == '__main__':
    path = './data.csv'
    data = pd.read_csv(path, encoding='utf-8')
    print(data.info())
    #bert = BertModel.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese", return_dict=False)
    tokenizer = BertTokenizer.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese")
    label = data['label'].values
    print(f'标签是{label}, {label.shape}')
    sen_len = sorted([len(tokenizer.encode(i)) for i in data['review']], reverse=True)
    print(f'长度排序{sen_len[:6]}, {type(sen_len)}')
    # 编码后句子最长为 458
    max_len = max(sen_len)
    data = [tokenizer.encode(i, max_length=max_len, padding='max_length') for i in data['review']]
    data = np.stack(data, axis=0)  # [batch, 458]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label, test_size=0.4, stratify=label)
    print(f'训练集大小：{x_train.shape},{y_train.shape}, 训练集正样本数{sum(y_train)}')
    print(f'测试集大小：{x_test.shape}, {y_test.shape}, 测试集正样本数{sum(y_test)}')
    model_type = 'bert'
    lr = 0.01
    batch = 1
    hidden_size = 256
    model = MyModel(hidden_size=hidden_size, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    train_loss_epoch = []
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train.astype(np.long)), torch.from_numpy(y_train.astype(np.long).reshape(-1, 1))), batch_size=batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test.astype(np.long)),
                                            torch.from_numpy(y_test.astype(np.long).reshape(-1, 1))), batch_size=batch,
                              shuffle=True, drop_last=True)
    for epoch in range(2):
        loss = train(train_loader, model, optimizer)
        train_loss.extend(loss)
        train_loss_epoch.append(sum(loss) / len(loss))
        print(f'epoch {epoch + 1:3d}, Average loss {sum(loss) / len(loss):.3f}')
    acc, ti = test(test_loader, model)
    print(f'使用的模型是：{model_type}，使用的学习率：{lr}，使用的隐藏单元大小：{hidden_size}，准确率：{acc}，测试 100条消耗时间{ti}')



