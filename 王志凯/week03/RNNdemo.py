import torch
import torch.nn as nn
import string
import random
import numpy as np
import json

"""
使用RNN判断某个字符在文本中的位置
判断样本类别：字母a在文本中出现的位置即为样本类别
如"bcadt"类别为[0,0,1,0,0]
"""

# 构建词表
def build_vocab():
    vocab_str = string.ascii_lowercase
    vocab = {v: k + 1 for k, v in enumerate(vocab_str)}
    vocab["[pad]"] = 0
    vocab["[unk]"] = len(vocab)
    return vocab


# 将文本映射到字典，长度不够补0，多余长度截断
def str_to_seq(text_str, vocab, sent_len):
    seq = [vocab.get(v, vocab["[unk]"]) for v in text_str][:sent_len]
    if len(seq) < sent_len:
        seq += [vocab["[pad]"]] * (sent_len - len(seq))
    return seq


# 随机构建多组字母组成的文本
# batch_size：生成多少条文本
# sent_len：一条文本长度
def build_text(data_size, sent_len, vocab):
    X = []
    text_array = []
    trans_vocab = {v: k for k, v in vocab.items()}
    for i in range(data_size):
        # 随机取text_len个不同的数字，防止取到pad和unk，范围为1-53
        num_array = random.sample(range(1, len(vocab) - 1), sent_len)
        # 数字转字母
        text = [trans_vocab[num] for num in num_array]
        text_array.append("".join(text))
        X.append(num_array)
    return text_array, torch.tensor(X)


# 构建训练数据
# 判断样本类别：字母a在文本中出现的位置即为样本类别
# 如"bcad"类别为[0,0,1,0]
def build_train_datas(X, vocab):
    Y = torch.zeros_like(X, dtype=torch.float32)
    for x, y in zip(X, Y):
        if vocab["a"] in x:
            a_index = (x == vocab["a"]).argwhere()
            y[a_index[0]] = 1
    return X, Y


# 检测模型
def predict(model_path, X_letter):
    sent_len = 5
    num_feature = 10
    hidden_size = 5
    # 将文本映射为数字
    vocab = build_vocab()
    X = [str_to_seq(x, vocab, sent_len) for x in X_letter]
    X = torch.tensor(X)
    # 初始化模型
    model = RNNModel(sent_len, num_feature, hidden_size, vocab)
    model.load_state_dict(torch.load(model_path))
    # 判断样本类型
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(X)
        torch.set_printoptions(sci_mode=False)
        for x, y in zip(X, y_pred):
            y_max_idx = torch.argmax(y)
            if x[y_max_idx] == vocab["a"]:
                print(f"输入为：{x}，输出为：{y}，预测正确")
            else:
                print(f"输入为：{x}，输出为：{y}，预测错误！！！")


class RNNModel(nn.Module):
    # sent_len：一句话长度
    # num_feature：每个字母特征数，即embedding_dim（需要将一个字母转化为多少向量）
    # hiden_size：RNN隐藏状态长度
    # vocab：词表
    def __init__(self, sent_len, num_feature, hidden_size, vocab):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), num_feature, padding_idx=0)
        # 对于文本处理，RNN属于天然的“池化层”，所以不需要进行池化
        self.RNN_layer = nn.RNN(num_feature, hidden_size, batch_first=True)
        self.LN = nn.LayerNorm(hidden_size)
        self.Linear = nn.Linear(hidden_size, sent_len)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, x = self.RNN_layer(x)
        x = self.LN(x.squeeze())
        y_pred = self.Linear(x)
        if y is None:
            return torch.softmax(y_pred, dim=1)
        return self.loss(y_pred, y)


def main():
    round_num = 20  # 训练总轮数
    data_size = 1000  # 总样本数
    batch_size = 20  # 一批样本个数
    sent_len = 5
    num_feature = 10
    hidden_size = 5
    lr = 0.01

    # 构建词表
    vocab = build_vocab()
    # 生成文本
    text, X = build_text(data_size, sent_len, vocab)
    # 通过文本构建训练集
    X, Y = build_train_datas(X, vocab)

    # 初始化模型
    model = RNNModel(sent_len, num_feature, hidden_size, vocab)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    model.train()
    for i in range(round_num):
        loss_round = []
        # 样本总批次
        batch_count = data_size // batch_size
        for batch in range(batch_count):
            batch_x = X[batch_size * batch:batch_size * (batch + 1)]
            batch_y = Y[batch_size * batch:batch_size * (batch + 1)]
            loss = model(batch_x, batch_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            loss_round.append(loss.item())
        print(f"第{i + 1}轮训练，该轮平均loss值为：{np.mean(loss_round)}")
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as file:
        file.write(json.dumps(vocab))


if __name__ == "__main__":
    # main()

    # 测试模型
    X_letter = ["abhcd", "balcd", "hhjaz", "sdfab", "warty"]
    predict("model.pt", X_letter)