# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的RNN网络编写,实现一个文本分类任务：
1. 包含"我"字的文本，按"我"字出现的位置分为5类（第0-4类）
2. 不包含"我"字的文本分为第5类
3. 输入文本长度不等于5个字符时，输出"无法预测"

"""


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128, num_layers=2):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)  # RNN层
        self.classify = nn.Linear(hidden_size, 6)  # 线性层，输出6个类别
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        # 取最后一个时间步的输出作为序列表示
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(last_output)  # (batch_size, hidden_size) -> (batch_size, 6)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 输出预测概率分布


# 构建字符集，使用常见中文字符
def build_vocab():
    # 使用常见中文字符，确保有足够的字符用于生成随机文本
    chars = "我爱中国家人朋友你好他们学习工作生活城市乡村山水风景美丽快乐幸福梦想希望" + \
            "abcdefghijklmnopqrstuvwxyz0123456789"  # 添加一些英文字母和数字增加多样性
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 从字表中选取sentence_length个字，构建随机文本
    available_chars = [char for char in vocab.keys() if char not in ['pad', 'unk']]
    x = [random.choice(available_chars) for _ in range(sentence_length)]

    # 检查是否包含"我"字，并确定其位置
    if "我" in x:
        # 找到"我"字第一次出现的位置（从0开始计数）
        position = x.index("我")
        # 位置从0-4对应类别0-4
        y = position
    else:
        # 不包含"我"字，属于第5类
        y = 5

    # 将字符转换为索引
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 直接存储标量，而不是列表
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = RNNModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)

    # 统计每个类别的样本数量
    class_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for label in y:
        class_count[int(label)] += 1

    print("各类别样本数量:")
    for i in range(0, 6):
        print(f"第{i + 1}类: {class_count[i]}个")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回概率分布
        predicted_classes = torch.argmax(y_pred, dim=1)  # 取概率最大的类别

        for y_p, y_t in zip(predicted_classes, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 64  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    learning_rate = 0.001  # 学习率

    # 建立字表
    vocab = build_vocab()
    print(f"字符表大小: {len(vocab)}")

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 64  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    # 检查输入字符串长度
    valid_strings = []
    invalid_strings = []

    for s in input_strings:
        if len(s) != sentence_length:
            invalid_strings.append(s)
        else:
            valid_strings.append(s)

    # 处理有效字符串
    x = []
    for input_string in valid_strings:
        # 将输入序列化，处理未知字符
        x.append([vocab.get(char, vocab['unk']) for char in input_string])

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        if x:  # 如果有有效输入
            result = model.forward(torch.LongTensor(x))  # 模型预测
        else:
            result = []

    # 输出无效字符串的结果
    for s in invalid_strings:
        print(f"输入：{s}, 预测结果：无法预测（文本长度不等于5个字符）")

    # 输出有效字符串的预测结果
    for i, input_string in enumerate(valid_strings):
        predicted_class = torch.argmax(result[i]).item()  # 获取预测类别 (0-5)
        probability = result[i][predicted_class].item()  # 获取对应类别的概率

        # 解释预测结果
        if predicted_class == 5:
            explanation = "不包含'我'字"
            display_class = 6  # 显示为第6类
        else:
            explanation = f"'我'字出现在第{predicted_class + 1}位"
            display_class = predicted_class + 1  # 显示为第1-5类

        print(f"输入：{input_string}, 预测类别：第{display_class}类, 概率值：{probability:.4f}, {explanation}")


if __name__ == "__main__":
    main()

    # 测试学习效果
    test_strings = [
        "我爱学习",
        "你喜欢我",
        "他们爱我",
        "我爱中国",
        "朋我友你好",
        "学习我快乐",
        "工作努我力",
        "幸福我生活",
        "我梦想飞翔",
        "城市我乡村",
        "美丽风景",
        "快乐家庭",
        "好好学习",
        "你好她爱我",
        "我",
        "你好",
        "我爱学习编程",
        "我非常喜欢学习",
        "你好吗我今天很高兴"
    ]

    print("\n" + "=" * 50)
    print("测试学习效果:")
    predict("rnn_model.pth", "vocab.json", test_strings)
