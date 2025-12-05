import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from text_preprocessor import TextPreprocessor
from sklearn.metrics import classification_report, accuracy_score
import jieba

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class TextDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return sequence, label

class CNNTextClassifier(nn.Module):
    """CNN文本分类器模型"""
    def __init__(self, vocab_size, embedding_dim=100, num_filters=128, filter_sizes=[3, 4, 5], dropout=0.5):
        super(CNNTextClassifier, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        
        # Sigmoid激活函数（二分类）
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_length, embedding_dim)
        
        # 卷积 + 池化
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        
        # 拼接特征
        cat = self.dropout(torch.cat(pooled, dim=1))  # (batch_size, len(filter_sizes) * num_filters)
        
        # 全连接层
        output = self.fc(cat)  # (batch_size, 1)
        
        # Sigmoid激活
        output = self.sigmoid(output)  # (batch_size, 1)
        
        return output

class CNNClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor(max_words=10000, max_length=100)
        self.model = None
        self.criterion = None
        self.optimizer = None
    
    def train(self, train_csv_path, batch_size=64, epochs=10, embedding_dim=100):
        """训练CNN分类器"""
        print("=== CNN文本分类器训练 ===")
        
        # 预处理训练数据
        print("预处理训练数据...")
        train_sequences, train_labels = self.preprocessor.preprocess_data(train_csv_path, fit_vocab=True)
        
        # 创建数据集和数据加载器
        train_dataset = TextDataset(train_sequences, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        vocab_size = self.preprocessor.vocab_size
        self.model = CNNTextClassifier(vocab_size, embedding_dim=embedding_dim).to(device)
        
        # 定义损失函数和优化器
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 训练模型
        print("开始训练...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                # 前向传播
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item() * sequences.size(0)
                predictions = (outputs >= 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_samples += sequences.size(0)
            
            # 计算平均损失和准确率
            epoch_loss = total_loss / total_samples
            epoch_acc = correct_predictions / total_samples
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        print("训练完成！")
        return
    
    def evaluate(self, test_csv_path, batch_size=64):
        """在测试集上评估模型"""
        print(f"\n=== 在{test_csv_path}上评估 ===")
        
        # 预处理测试数据
        test_sequences, test_labels = self.preprocessor.preprocess_data(test_csv_path, fit_vocab=False)
        
        # 创建数据集和数据加载器
        test_dataset = TextDataset(test_sequences, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 评估模型
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                # 前向传播
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item() * sequences.size(0)
                predictions = (outputs >= 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_samples += sequences.size(0)
                
                # 保存预测结果和真实标签
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        # 转换为一维数组
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # 打印分类报告
        print(f"损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(all_labels, all_predictions, target_names=['负样本', '正样本']))
        
        return accuracy
    
    def predict(self, text):
        """预测单个文本的类别"""
        # 分词
        tokens = jieba.lcut(text)
        
        # 转换为序列
        sequence = [self.preprocessor.word_index.get(token, 1) for token in tokens]
        
        # Padding
        padded_sequence = np.zeros(self.preprocessor.max_length, dtype=np.int32)
        length = min(len(sequence), self.preprocessor.max_length)
        padded_sequence[:length] = sequence[:length]
        
        # 转换为张量
        sequence_tensor = torch.tensor(padded_sequence, dtype=torch.long).unsqueeze(0).to(device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(sequence_tensor)
            predicted_label = (output >= 0.5).float().item()
        
        return int(predicted_label)

def main():
    # 创建分类器实例
    cnn_classifier = CNNClassifier()
    
    # 训练模型
    cnn_classifier.train("train.csv", batch_size=64, epochs=10, embedding_dim=100)
    
    # 评估模型
    print("\n=== 在测试集上评估 ===")
    cnn_classifier.evaluate("test.csv")
    
    print("\n=== 在验证集上评估 ===")
    cnn_classifier.evaluate("valid.csv")
    
    # 测试单个预测
    print("\n=== 测试单个预测 ===")
    sample_texts = [
        "这个餐厅的食物非常好吃，服务也很周到！",
        "这个餐厅的食物很难吃，服务也很差！"
    ]
    
    for text in sample_texts:
        predicted_label = cnn_classifier.predict(text)
        print(f"文本：{text}")
        print(f"预测类别：{'正样本' if predicted_label == 1 else '负样本'}")

if __name__ == "__main__":
    main()