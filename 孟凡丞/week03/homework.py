import random
import string

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

"""

基于pytorch的网络编写
判断特定文本字符在文本中的位置

"""

ALL_CHARS = list(string.ascii_lowercase) + [' ']

class TorchModel(nn.Module):
    def __init__(self, vocab_size, vector_dim=32, hidden_dim=64, num_layers=1, dropout=0.3):
        super(TorchModel, self).__init__()
        # 嵌入层Embedding
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        # LSTM层
        self.lstm = nn.LSTM(vector_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层
        self.classify = nn.Linear(hidden_dim, 1)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 交叉熵
        self.loss = nn.BCEWithLogitsLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.embedding(x)
        # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_dropped = self.dropout(lstm_out)
        logits = self.classify(lstm_dropped).squeeze(-1)
        if y is not None:
            return self.loss(logits, y)
        else:
            return torch.sigmoid(logits)


# 构建词汇表
def build_vocab():
    vocab = {"pad": 0}
    for index, char in enumerate(ALL_CHARS):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 构建数据集
def build_dataset(vocab, target_char, sentence_length, train_sample_num):
    X = []
    Y = []
    non_target_chars = [c for c in ALL_CHARS if c != target_char]
    for _ in range(train_sample_num):
        sampled_chars = []
        labels = []
        # 每个位置随机决定是否为目标字符
        for _ in range(sentence_length):
            if random.random() < 0.2:  # 20% 概率放置目标字符
                sampled_chars.append(target_char)
                labels.append(1.0)
            else:
                sampled_chars.append(random.choice(non_target_chars))
                labels.append(0.0)
        # 转换为索引
        input_indices = [vocab[char] for char in sampled_chars]
        # 生成标签：如果字符 == target_char，则为1，否则为0
        X.append(input_indices)
        Y.append(labels)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.long)  # (train_sample_num, sentence_length)
    Y = torch.tensor(Y, dtype=torch.float)  # (train_sample_num, sentence_length)
    return X, Y


# 测试函数
def evaluate_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_positions = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            y_pred = model(batch_x)  # shape: (batch_size, seq_len)
            predicted_labels = (y_pred > 0.5).float()  # 转为 0/1
            # 累加统计
            correct_predictions += (predicted_labels == batch_y).sum().item()
            total_positions += batch_y.numel()
    # 计算整体准确率
    accuracy = correct_predictions / total_positions
    print(f"\n======== 测试结果 ========")
    print(f"测试样本数: {len(dataloader.dataset)}")
    print(f"总预测位置数: {total_positions}")
    print(f"正确预测位置数: {int(correct_predictions)}")
    print(f"准确率: {accuracy:.6f}")


# 训练函数
def train_model(model, optimizer, num_epochs, batch_size, train_sample_num, test_sample_num, vocab, target_char,
                sentence_length):
    # 构建数据集
    x, y = build_dataset(vocab, target_char, sentence_length, train_sample_num)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 计算loss
            loss = model(batch_x, batch_y)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
            # 累加批次loss
            total_loss += loss.item()
        print(f"=========\n第{epoch + 1}轮平均loss: {total_loss / len(train_loader):.6f}")
        # 测试，每次生成新的测试集
        # 测试集
        x, y = build_dataset(vocab, target_char, sentence_length, test_sample_num)
        dataset = TensorDataset(x, y)
        evaluate_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        evaluate_model(model, evaluate_loader)


if __name__ == '__main__':
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 10  # 每批训练样本个数
    learning_rate = 0.001  # 学习率
    total_sample_num = 1000  # 样本总数
    test_sample_num = 500  # 测试集样本总数
    vocab = build_vocab()  # 构建词汇表
    vocab_size = len(vocab)  # 词汇表长度
    target_char = 'w'  # 目标字符
    sentence_length = 100  # 句子长度

    # 建立模型
    model = TorchModel(vocab_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练
    train_model(model, optimizer, epoch_num, batch_size, total_sample_num, test_sample_num, vocab, target_char,
                sentence_length)
