import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
import random
#coding:utf8


# 定义分类模型
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)
        self.classify = nn.Linear(hidden_size, sentence_length + 1)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, y=None):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        x = output[:, -1, :]
        y_pred = self.classify(x)
        if y is not None:
            loss = self.loss(y_pred, y)
            return loss
        else:
            return y_pred
# 定义词汇表
def build_vocab():
    chars = "你我他adefghijklmnopqrstuvwxyz"
    vocab = {'pad': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

# 随机生成样本
def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    if 'a' in x:
        y = x.index('a')
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]

    return x, y #返回一个句子和它的类别




# 建立数据集

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型

def build_model(vocab, char_dim, sentence_length, hidden_size):
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)
    return model

# 测试代码

def evaluate(model, vocab,sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    print(f'本次验证集中共有{len(y)}个样本')
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print(f'预测正确个数：{correct}, 正确率：{correct/(correct+wrong):.5f}')
    return correct / (correct + wrong)
def main():
    # 配置参数
    epoch_num = 20
    batch_size = 40
    train_sample = 1000
    char_dim = 128
    sentence_length = 10
    lr = 0.001
    hidden_size = 256
    #建立字表
    vocab = build_vocab()
    #建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 128
    sentence_length = 10
    vocab = json.load(open(vocab_path, 'r', encoding="utf8"))
    hidden_size = 256
    model = build_model(vocab, char_dim, sentence_length, hidden_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(f'输入：{input_string}，预测类别：{torch.argmax(result[i])},预测概率值：{nn.functional.softmax(result[i], dim=0).max()}')


if __name__ == '__main__':
    main()
    test_strings = ['kijabcdefh', 'gijkbcdeaf', 'gkijadfbec', 'kijhdefacb']
    predict("model.pth", "vocab.json", test_strings)





