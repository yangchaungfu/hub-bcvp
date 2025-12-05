"""
8个字符随机生成5个字然后通过rnn网络进行分辨我这个字在哪一位，使用交叉熵
"""




import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# =========================
# 字符表与编码
# =========================
char2idx = {
    '你': 1,
    '我': 2,
    '他': 3,
    '中': 4,
    '国': 5,
    '大': 6,
    '家': 7,
    '好': 8,
    '0': 0,   # padding
    '9': 9    # unknown
}
idx2char = {v: k for k, v in char2idx.items()}
vocab_size = len(char2idx)

# =========================
# RNN 模型定义
# =========================
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步输出
        return out

# =========================
# 随机生成训练数据
# =========================
def generate_random_data(num_samples=500):
    sentences = []
    labels = []

    chars = list(char2idx.keys())
    chars.remove('0')
    chars.remove('9')

    for _ in range(num_samples):
        length = 5
        s = ''.join(random.choice(chars) for _ in range(length))
        if '我' in s:
            position = s.index('我')
            label = position + 1  # 1~5类
        else:
            label = 6  # 没有“我” → 第6类
        sentences.append(s)
        labels.append(label - 1)  # 从0开始
    return sentences, labels

# =========================
# 数据编码
# =========================
def encode_sentence(sentence):
    seq = [char2idx.get(ch, 9) for ch in sentence]
    while len(seq) < 5:
        seq.append(0)
    return torch.tensor(seq[:5], dtype=torch.long)

# =========================
# 模型训练函数
# =========================
def train_model(model, optimizer, num_epochs=200, patience=5):
    sentences, labels = generate_random_data()
    inputs = torch.stack([encode_sentence(s) for s in sentences])
    targets = torch.tensor(labels)

    prev_loss = None
    stable_count = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == targets).float().mean().item()

        print(f"第 {epoch + 1} 轮训练: 准确率 = {acc:.3f}, 损失 = {loss.item():.4f}")

        # ---- loss稳定即退出 ----
        if prev_loss is not None:
            if abs(prev_loss - loss.item()) < 1e-4:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= patience:
                print(f"✅ Loss稳定，提前停止训练 at epoch {epoch + 1}")
                break
        prev_loss = loss.item()

# =========================
# 主函数入口
# =========================
def main():
    model = RNNClassifier(input_size=vocab_size, hidden_size=32, output_size=6)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, optimizer)

    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/rnn_char_classify.pth")
    print("模型已保存至 models/rnn_char_classify.pth")

    # 测试部分（已并入主函数）
    model.eval()
    test_cases = [
        "我中国大家",   # 第1类
        "你我中国好",   # 第2类
        "我爱你啊",   # 第3类
        "你他中国吗",   # 第5类
        "你好大家中"    # 第6类
    ]

    print("\n================ 测试结果 ================")
    for s in test_cases:
        x = encode_sentence(s).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item() + 1
        print(f"输入：{s} → 预测类别：第{pred}类")


if __name__ == "__main__":
    main()
