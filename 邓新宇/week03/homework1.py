import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchRNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)    #Embedding
        self.layers = nn.RNN(vector_dim, hidden_size, batch_first=True)    #RNN层
        self.classify = nn.Linear(hidden_size, sentence_length+1)    #线性层
        self.loss = nn.CrossEntropyLoss()    #交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)    #先过Embedding层 (batch_size, sentence_length) -> (batch_size, sentence_length, vector_dim)
        out, hn = self.layers(x)    #out: (batch_size, sentence_length, hidden_size)
        hn = hn.squeeze()           #hn: (batch_size, hidden_size)
        y_pred = self.classify(hn)  #(batch_size, hidden_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab["unk"] = len(vocab)
    return vocab

#随机生成样本
#语序有关的多分类任务，句子中“我”在哪个下标就属于哪一类，没在句子中也为一类
#若出现重复，则第一个“我”在哪个下标就属于哪一类
def build_sample(vocab, sentence_length):
    normal_words = [word for word in vocab.keys() if word != "pad"]
    has_me = random.random() < 0.8    #一句话中80%的概率出现“我”
    core_length = random.randint(1, sentence_length)  # 核心内容至少1个词，避免空句
    core_words = []
    first_me_pos = -1    #“我”随机出现的下标

    if has_me:    #句子中有“我”
        first_me_pos = random.randint(0, core_length - 1)  # 随机一个“我”的位置
        for i in range(core_length):
            if i == first_me_pos:
                core_words.append("我")
            else:
                core_words.append(random.choice(normal_words))
    else:    #句子中无“我”
        other_words = [word for word in normal_words if word != "我"]
        core_words = [random.choice(other_words) for _ in range(core_length)]

    #填充pad：核心内容长度不足时，仅在末尾补pad（符合pad的用途）
    x = core_words + ["pad"] * (sentence_length - core_length)

    me_positions = [idx for idx, word in enumerate(x) if word == "我"]
    if me_positions:  # 有“我”，标签=实际首次位置
        label = me_positions[0]
    else:  # 无“我”，标签=句子长度
        label = sentence_length

    return x, label

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        x_idx = [vocab[word] for word in x]  # 每个词 -> 对应的整数索引
        dataset_x.append(x_idx)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length, hidden_size=128):
    model = TorchRNNModel(char_dim, sentence_length, vocab, hidden_size)
    return model

def evaluate_model(model, vocab, sentence_length):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num, vocab, sentence_length)
    with torch.no_grad():
        y_pred = model(x)
        prob_dist = torch.softmax(y_pred, dim=1)
        pred_indices = y_pred.argmax(dim=1)
        correct = (pred_indices == y).sum().item()
        accuracy = correct / test_sample_num
        # 打印每个样本的真实五维数据、真实类别、概率分布和预测类别（前5个示例）
        print("\n部分样本的预测详情（真实数据 | 真实类别 | 概率分布 | 预测类别）：")
        for i in range(min(5, test_sample_num)):  # 只打印前5个，避免输出过长
            true_data = x[i].numpy()  # 第i个样本的真实数据（转为numpy数组）
            true_label = y[i].item()  # 真实类别（0-8）
            probs = prob_dist[i].numpy()  # 当前样本的9类概率
            pred_label = pred_indices[i].item()  # 预测类别（0-8）

            # 格式化打印：所有数值保留3位小数，保持格式统一
            print(
                f"样本{i + 1}：数据=[{true_data[0]:.3f}, {true_data[1]:.3f}, {true_data[2]:.3f}, {true_data[3]:.3f}, {true_data[4]:.3f}, {true_data[5]:.3f}, {true_data[6]:.3f}, {true_data[7]:.3f}] "
                f"| 真实={true_label} "
                f"| 概率=[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}, {probs[3]:.3f}, {probs[4]:.3f}, {probs[5]:.3f}, {probs[6]:.3f}, {probs[7]:.3f}, {probs[8]:.3f}] "
                f"| 预测={pred_label}"
            )

        print(f"\n正确预测个数：{correct}, 正确率：{accuracy:.4f}")
        return accuracy

def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, result[i].argmax().item())) #打印结果

def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length, hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate_model(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

if __name__ == "__main__":
    main()
    test_strings = ["f我vf我e我d", "我z你dfgad", "我qwde我ge", "n我kwwws我"]
    predict("model.pth", "vocab.json", test_strings)




