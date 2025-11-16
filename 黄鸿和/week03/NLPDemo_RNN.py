#coding:utf8
# 时间：2025-10-30
# 作者：黄鸿和

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于 PyTorch 的 RNN 版本，完成“位置分类”任务：
- 随机生成长度为 L 的序列；以 50% 概率在随机位置放入目标字符（“黄/鸿/和”）；否则不放入。
- 标签 y 为位置索引（0..L-1）；若未出现，则标签为 L（表示“未出现”）。

模型结构：Embedding → RNN → 线性层（输出 L+1 类）
训练损失：CrossEntropyLoss（输入 logits，目标为整数类索引）
激活函数：最后使用 Softmax 归一化，输出 L+1 类概率

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, rnn_hidden=64):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=rnn_hidden,
            nonlinearity='tanh',
            batch_first=True
        )
        hidden_out = rnn_hidden
        self.num_classes = sentence_length + 1              # 0..L-1 位置 + L 表示未出现
        self.classify = nn.Linear(hidden_out, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值(概率)
    def forward(self, x, y=None):
        x = self.embedding(x)                      # (batch, sen_len) -> (batch, sen_len, vector_dim)
        output, h_n = self.rnn(x)                  # h_n: (num_layers*num_directions, batch, hidden)
        last_hidden = h_n[-1]                      # (batch, hidden_out)
        logits = self.classify(last_hidden)        # (batch, num_classes)
        if y is not None:
            return self.loss_fn(logits, y)         # CrossEntropyLoss，y 为类别索引 LongTensor
        else:
            return torch.softmax(logits, dim=-1)   # 预测时返回每类概率

# 字符集（包含目标字符与其他字母），并保留 pad/unk
def build_vocab():
    chars = "黄鸿和defghijklmnopqrst"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab

# 随机生成一个样本：
# - 以 50% 概率在随机位置放入“黄/鸿/和”之一，标签为该位置索引（0..L-1）
# - 否则不放入目标字符，标签为 L（表示未出现）
def build_sample(vocab, sentence_length):
    targets = set(["黄", "鸿", "和"]) 
    all_chars = [c for c in vocab.keys() if c not in ("pad", "unk")]
    non_targets = [c for c in all_chars if c not in targets]
    x = [random.choice(non_targets) for _ in range(sentence_length)]
    if random.random() < 0.5:
        pos = random.randrange(sentence_length)      # 随机选择一个位置放置目标字符
        x[pos] = random.choice(list(targets))
        y = pos                                      # 位置作为标签（0..L-1）
    else:
        y = sentence_length                          # 未出现用 L 表示
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    num_none = int((y == x.shape[1]).sum())
    print("测试集中‘未出现’样本：%d, ‘出现’样本：%d" % (num_none, len(y) - num_none))
    correct, wrong = 0, 0
    with torch.no_grad():
        prob = model(x)                            # (batch, L+1)
        pred = prob.argmax(dim=-1)                 # 取概率最大的类别索引
        correct = int((pred == y).sum())
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    # 建立字表/模型
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 画出准确率与Loss在同一张图
    epochs = list(range(1, epoch_num + 1))
    acc_list = [item[0] for item in log]
    loss_list = [item[1] for item in log]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    l1, = ax1.plot(epochs, loss_list, color='tab:blue', marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:red')
    l2, = ax2.plot(epochs, acc_list, color='tab:red', marker='s', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    plt.title('Training Curve: Accuracy & Loss')
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close(fig)

    # 保存
    torch.save(model.state_dict(), "model_rnn.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        ids = [vocab.get(char, vocab['unk']) for char in input_string]
        if len(ids) < sentence_length:
            ids = ids + [vocab.get('pad', 0)] * (sentence_length - len(ids))  # 右侧padding到固定长度
        else:
            ids = ids[:sentence_length]  # 截断到固定长度
        x.append(ids)
    model.eval()
    with torch.no_grad():
        prob = model.forward(torch.LongTensor(x))          # (batch, L+1)
        # 取概率最大的类别索引
        pred = prob.argmax(dim=-1)
    for i, input_string in enumerate(input_strings):
        idx = int(pred[i])
        # 取概率最大的类别索引对应的概率
        p = float(prob[i, idx])
        # idx为句子长度，表示未出现
        tag = "未出现" if idx == sentence_length else f"位置{idx}"
        print("输入：%s, 预测：%s, 置信度：%.4f" % (input_string, tag, p))


if __name__ == "__main__":
    main()
    # 测试样本长度都是 6
    test_strings = [
    "黄abcde",  # 位置0
    "a鸿cdef",  # 位置1
    "ab和def",  # 位置2
    "abc黄ef",  # 位置3
    "abcd鸿f",  # 位置4
    "abcde和",  # 位置5
    "abcdef",   # 未出现
    "a和cdef",
    "ab鸿def",
    "abcd和f",
    "和bcdef",
    "a黄bcde",
    "abcedf",   # 未出现
    "h黄ijkl",
    "mno鸿qr",
    "st和uvw",
    "x和zabc",
    "de黄ghi",
    "jklmno",   # 未出现
    "pq鸿stu",
    "vwxyz和",
    "a和cde",   # 短 -> 会padding
    "ab黄e",    # 短 -> 会padding
    "zzzz和gh", # 长 -> 会截断
    ]
    predict("model_rnn.pth", "vocab.json", test_strings)


