# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
实现一个nlp任务：给定长度为len的文本，判定关键词在文本中的位置，根据最早出现的位置将本文分类。
    如果关键字最早出现在位置i，那么将该本文归为i类(0<=i<len)；如果没出现，就归为len类。 
    比如，文本 "哈哈，我回到祖国啦，我很开心！"，指定词为："我"、"你"、"他"，因为第3位最早出现了"我"，所以这个文本应归为第3类。
"""


class TorchModel(nn.Module):
    def __init__(self, sentence_len, token_dim, hidden_size, vocab):
        """
        Args:
            sentence_len: 单个文本的长度
            token_dim: 每个词对应的向量维度
            hidden_size:
            vocab: 词表
        """
        # print("vocab:\n", vocab)
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), token_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(token_dim, hidden_size, bias=False, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentence_len) -> (batch_size, sentence_len, token_dim)
        x, _ = self.rnn(x)  # (batch_size, sentence_len, token_dim) -> (batch_size, sentence_len, hidden_size)
        y_pred = self.linear(x)  # (batch_size, sentence_len, hidden_size) -> (batch_size, sentence_len, 1)
        y_pred = y_pred.squeeze(-1)  # (batch_size, sentence_len, 1) -> (batch_size, sentence_len)

        # print("y_pred: ", y_pred)
        # print("y_true: ", y)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失, 注意：y是真实标签, shape为(batch_size), 这样才符合交叉熵Loss的传参要求
        else:
            return y_pred  # 输出预测结果


# 构建词表， "abc" -> {"pad": 0, "a":1, "b":2, "c":3, "unk":4}
def build_vocab():
    chars = "你好，朋友！Go我和he他de早rqw3g"  # 字符集, 不得包含重复字符, 否则embedding时会报错。
    vocab = {"[pad]": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab["[unk]"] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 从词表随机选取sentence_length个词
    sentence = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    sentence.append("[pad]")  # 在末位扩充一个无意义的词，以适配“关键字不在文本中” 的len分类

    # 如果关键字最早出现在位置i，那么将该本文归为i类(0<=i<len)；如果没出现，就归为len类。
    keywords = ['我', '你', '他']
    min_pos = len(sentence) - 1
    for keyword in keywords:
        try:
            pos = sentence.index(keyword)
            if pos < min_pos:
                min_pos = pos
        except ValueError as e:
            continue
    return sentence, min_pos


# 建立数据集
def build_dataset(total_sample_num, vocab, sentence_len):
    dataset_x = []
    dataset_y = []
    for _ in range(total_sample_num):
        x, y = build_sample(vocab, sentence_len)
        x = [vocab.get(w, vocab["[unk]"]) for w in x]  # 将字转换成序号，为了做embedding
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 测试、评估模型的准确率
def evaluate(model, vocab, sentence_len):
    model.eval()  # 设置模型为评估模式，等同于 model.train(False)
    test_sample_num = 200
    test_x, test_y = build_dataset(test_sample_num, vocab, sentence_len)

    stat = np.zeros(sentence_len + 1)
    for label in test_y:
        stat[label] += 1

    for i in range(len(stat)):
        print(f"本次预测集有{int(stat[i])}个{i}类样本")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(test_x)  # 模型预测 model.forward(x)
        for y_pred, y_true in zip(y_pred, test_y):
            if torch.argmax(y_pred) == y_true:
                correct += 1
            else:
                wrong += 1
        print(f"正确预测个数：{correct}，正确率：{correct * 100 / (correct + wrong)}%")
        return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 12  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample_num = 600  # 每轮训练总共训练的样本总数
    learning_rate = 0.005  # 学习率

    sentence_len = 10  # 单个文本的长度
    token_dim = 8  # 每个词对应的向量维度
    hidden_size = 16  # 隐藏层维度

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(sentence_len, token_dim, hidden_size, vocab)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample_num / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_len)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        avg_loss = round(np.mean(watch_loss), 4)  # 本轮的平均Loss
        print(f"=========\n第{epoch + 1}轮的平均loss: {avg_loss}")

        accuracy = evaluate(model, vocab, sentence_len)  # 测试本轮模型结果
        log.append([accuracy, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "NLP_text_classify.pth")

    # 保存词表
    writer = open("NLP_text_classify_vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    # 画图
    print("\n每轮训练的准确率及损失:\n", log)

    print("\n绘制准确率及损失曲线...")
    plt.plot(range(len(log)), [item[0] for item in log], label="accuracy")  # 画accuracy曲线
    plt.plot(range(len(log)), [item[1] for item in log], label="loss")  # 画loss曲线
    plt.legend(title='深度学习-NLP-分本分类')

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统下中文可以使用'SimHei'或'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.show()
    return


def str_to_sequence(vocab, sentence, max_len):
    seq = [vocab.get(s, vocab["[unk]"]) for s in sentence[:max_len]]
    if len(seq) < max_len:
        seq += [vocab["[pad]"]] * (max_len - len(seq))
    seq += [vocab["[pad]"]]  # 在末位扩充一个无意义的词，以适配“关键字不在文本中” 的len分类
    return seq


# 使用训练好的模型做预测
def predict(model_path, vocab_path, sentences):
    sentence_len = 10  # 单个文本的长度
    token_dim = 8  # 每个词对应的向量维度
    hidden_size = 16  # 隐藏层维度

    # 加载词表
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表

    # 将句子转为模型入参
    input_matrix = []
    for sentence in sentences:
        # print(str_to_sequence(vocab, sentence, sentence_length))
        input_matrix.append(str_to_sequence(vocab, sentence, sentence_len))  # 将字转换成序号，为了做embedding

    # 建立模型
    model = TorchModel(sentence_len, token_dim, hidden_size, vocab)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print("model.state_dict():\n", model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(input_matrix))  # 模型预测

    for sentence, y in zip(sentences, result):
        print(f"输入：{sentence:10s}, 输出:{y.numpy().round(5)}, 预测类别：{torch.argmax(y)}\n")


if __name__ == "__main__":
    main()
    print("\n使用训练好的模型做预测...")
    print("     规则：如果关键字('你'、'我'、'他')出现在文本的位置i，那么将该本文归为i类(0<=i<10)；如果没出现，就归为10类。\n ")

    input_data = ["rqwde33g", "fn我是一个程序员", "哈哈，他说我不懂", "wz你dfga", "n只不akwww我"]
    predict("NLP_text_classify.pth", "NLP_text_classify_vocab.json", input_data)
