import pandas as pd
import jieba
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# ================= 配置区域 =================
INPUT_CSV = 'text_data.csv'  # 测试文件加载
OUTPUT_FILE = 'model_comparison_3.csv'
MAX_LEN = 128  # 文本截断长度
BATCH_SIZE = 16  # 批次大小
EPOCHS = 5  # 训练轮数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前运行设备: {DEVICE}")


# ===========================================

# --- 1. 数据预处理与 Dataset 定义 ---

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # 使用 jieba 分词
        words = list(jieba.cut(text))

        # 将词转换为ID，未知词填1 (<UNK>), 填充填0 (<PAD>)
        seq = [self.vocab.get(w, 1) for w in words]  # 1 is UNK

        # 截断或填充
        if len(seq) < MAX_LEN:
            seq = seq + [0] * (MAX_LEN - len(seq))  # 0 is PAD
        else:
            seq = seq[:MAX_LEN]

        return {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def build_vocab(texts, min_freq=1):
    """构建词表"""
    words = []
    for text in texts:
        words.extend(jieba.cut(str(text)))
    count = Counter(words)
    # 0 reserved for padding, 1 reserved for unknown
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in count.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


# --- 2. 模型定义 ---

# 模型 1: FastText
class FastTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, input_ids, **kwargs):
        embedded = self.embedding(input_ids)
        return self.fc(embedded)


# 模型 2: LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, **kwargs):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])


# 模型 3: Gated CNN
class GatedCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_linear = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_gate = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, **kwargs):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)
        A = self.conv_linear(embedded)
        B = self.conv_gate(embedded)
        H = A * torch.sigmoid(B)
        H = torch.max(H, dim=2)[0]
        return self.fc(H)


# --- 3. 辅助函数 ---

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def measure_speed(model, loader):
    model.eval()
    count = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            _ = model(input_ids=input_ids)
            count += input_ids.size(0)
            if count >= 100:
                break
    end_time = time.time()
    if count == 0: return 0
    actual_time = end_time - start_time
    return (actual_time / count) * 100


# --- 4. 主程序 ---

def main():
    print("正在读取数据...")
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(INPUT_CSV, encoding='gbk')
        except Exception as e:
            print(f"读取文件失败，请检查路径或编码: {e}")
            return

    if 'review' not in df.columns or 'label' not in df.columns:
        print("错误：CSV文件必须包含 'review' 和 'label' 两列")
        return

    df = df.dropna(subset=['review', 'label'])

    pos_size = len(df[df['label'] == 1])
    neg_size = len(df[df['label'] == 0])
    avg_len = df['review'].astype(str).apply(len).mean()

    print("正在拆分数据集 9:1 ...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'].values, df['label'].values, test_size=0.1, random_state=42
    )

    print("正在构建词表...")
    vocab = build_vocab(X_train)
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")

    models_config = [
        {"name": "fastText", "lr": 0.01, "hidden": 64, "builder": lambda: FastTextModel(vocab_size, 64, 2)},
        {"name": "LSTM", "lr": 0.001, "hidden": 128, "builder": lambda: LSTMModel(vocab_size, 64, 128, 2)},
        {"name": "GatedCNN", "lr": 0.001, "hidden": 100, "builder": lambda: GatedCNNModel(vocab_size, 64, 100, 2)}
    ]

    results = []

    for config in models_config:
        print(f"\n====== 正在训练模型: {config['name']} ======")
        train_ds = TextDataset(X_train, y_train, vocab=vocab)
        test_ds = TextDataset(X_test, y_test, vocab=vocab)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = config['builder']().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            start_t = time.time()
            train_epoch(model, train_loader, optimizer, criterion)
            print(f"Epoch {epoch + 1}/{EPOCHS} 完成, 耗时 {time.time() - start_t:.1f}s")

        acc = evaluate(model, test_loader)
        speed = measure_speed(model, test_loader)
        print(f"-> 准确率: {acc:.4f}")
        print(f"-> 推理速度: {speed:.4f} 秒/100条")

        results.append({
            "model_name": config['name'],
            "Learning_rate": config['lr'],
            "hidden_size": config['hidden'],
            "positive_sample_size": pos_size,
            "average_text_length": round(avg_len, 2),
            "negative_sample_size": neg_size,
            "Accuracy": round(acc, 4),
            "time_to_process_100_items": f"{speed:.4f} sec"
        })

    # --- 导出结果 ---
    print(f"\n正在保存结果到 {OUTPUT_FILE} ...")
    res_df = pd.DataFrame(results)
    cols = ["model_name", "Learning_rate", "hidden_size",
            "positive_sample_size", "average_text_length",
            "negative_sample_size", "Accuracy", "time_to_process_100_items"]
    res_df = res_df[cols]

    res_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("完成！")
    print(res_df)


if __name__ == "__main__":
    main()
