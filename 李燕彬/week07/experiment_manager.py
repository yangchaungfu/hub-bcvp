import os
import time
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jieba
from text_preprocessor import TextPreprocessor

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ExperimentManager:
    def __init__(self, output_file='experiment_results.csv'):
        self.output_file = output_file
        self.results = []
        
        # 初始化CSV文件
        self._init_csv()
    
    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                '模型类型',
                '实验次数',
                '学习率',
                '隐藏层大小',
                '训练损失值',
                '测试集准确率',
                '验证集每100条预测耗时(s)',
                '训练时间(s)'
            ])
    
    def add_result(self, model_type, exp_num, learning_rate, hidden_size, train_loss, test_accuracy, valid_time_per_100, train_time):
        """添加实验结果"""
        result = [
            model_type,
            exp_num,
            learning_rate,
            hidden_size,
            train_loss,
            test_accuracy,
            valid_time_per_100,
            train_time
        ]
        self.results.append(result)
        
        # 写入CSV
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(result)
    
    def run_svm_experiments(self, num_experiments=5):
        """运行SVM模型实验"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import SVC
        
        # 加载数据
        def load_data(file_path):
            df = pd.read_csv(file_path)
            return df['review'].tolist(), df['label'].tolist()
        
        train_texts, train_labels = load_data('train.csv')
        test_texts, test_labels = load_data('test.csv')
        valid_texts, valid_labels = load_data('valid.csv')
        
        # 分词处理
        print("SVM实验：分词处理中...")
        train_seg = [' '.join(jieba.lcut(text)) for text in train_texts]
        test_seg = [' '.join(jieba.lcut(text)) for text in test_texts]
        valid_seg = [' '.join(jieba.lcut(text)) for text in valid_texts]
        
        # 文本向量化
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_seg)
        X_test = vectorizer.transform(test_seg)
        X_valid = vectorizer.transform(valid_seg)
        
        # SVM超参数组合
        param_combinations = [
            {'C': 0.1, 'kernel': 'linear'},    # 实验1
            {'C': 1.0, 'kernel': 'linear'},    # 实验2
            {'C': 10.0, 'kernel': 'linear'},   # 实验3
            {'C': 1.0, 'kernel': 'rbf'},       # 实验4
            {'C': 10.0, 'kernel': 'rbf'}       # 实验5
        ]
        
        for i in range(min(num_experiments, len(param_combinations))):
            print(f"\nSVM实验 {i+1}/{num_experiments}")
            params = param_combinations[i]
            
            # 训练SVM
            start_time = time.time()
            svm = SVC(**params)
            svm.fit(X_train, train_labels)
            train_time = time.time() - start_time
            
            # 计算训练损失（SVM没有显式的训练损失，这里用0表示）
            train_loss = 0.0
            
            # 测试集准确率
            y_pred_test = svm.predict(X_test)
            test_accuracy = accuracy_score(test_labels, y_pred_test)
            
            # 验证集预测耗时
            start_time = time.time()
            y_pred_valid = svm.predict(X_valid)
            total_valid_time = time.time() - start_time
            valid_time_per_100 = (total_valid_time / len(valid_labels)) * 100
            
            # 记录结果
            self.add_result(
                model_type='SVM',
                exp_num=i+1,
                learning_rate=params['C'],  # SVM用C代替学习率
                hidden_size=0,  # SVM没有隐藏层大小
                train_loss=train_loss,
                test_accuracy=test_accuracy,
                valid_time_per_100=valid_time_per_100,
                train_time=train_time
            )
            
            print(f"实验 {i+1} 完成: C={params['C']}, kernel={params['kernel']}, 测试准确率={test_accuracy:.4f}")
    
    def run_bayes_experiments(self, num_experiments=5):
        """运行贝叶斯模型实验"""
        from bayes_text_classifier import BayesTextClassifier
        
        # 加载数据
        def load_data(file_path):
            df = pd.read_csv(file_path)
            return df
        
        train_data = load_data('train.csv')
        test_data = load_data('test.csv')
        valid_data = load_data('valid.csv')
        
        # 贝叶斯模型不需要超参数调整，运行多次以获取不同的训练结果
        for i in range(num_experiments):
            print(f"\n贝叶斯实验 {i+1}/{num_experiments}")
            
            # 初始化模型
            bayes = BayesTextClassifier()
            
            # 训练模型
            start_time = time.time()
            bayes.train(train_data)
            train_time = time.time() - start_time
            
            # 计算训练损失（贝叶斯模型没有显式的训练损失，这里用0表示）
            train_loss = 0.0
            
            # 测试集准确率
            test_accuracy, _, _, _ = bayes.evaluate(test_data)
            
            # 验证集预测耗时
            start_time = time.time()
            valid_accuracy, _, _, _ = bayes.evaluate(valid_data)
            total_valid_time = time.time() - start_time
            valid_time_per_100 = (total_valid_time / len(valid_data)) * 100
            
            # 记录结果
            self.add_result(
                model_type='贝叶斯',
                exp_num=i+1,
                learning_rate=0.0,  # 贝叶斯模型没有学习率
                hidden_size=0,  # 贝叶斯模型没有隐藏层
                train_loss=train_loss,
                test_accuracy=test_accuracy,
                valid_time_per_100=valid_time_per_100,
                train_time=train_time
            )
            
            print(f"实验 {i+1} 完成: 测试准确率={test_accuracy:.4f}")
    
    def run_cnn_experiments(self, num_experiments=5):
        """运行CNN模型实验"""
        from cnn_text_classifier import CNNClassifier, TextDataset, CNNTextClassifier
        
        # 定义超参数组合
        param_combinations = [
            {'learning_rate': 0.001, 'hidden_size': 128, 'embedding_dim': 100, 'kernel_sizes': [3, 4, 5], 'num_filters': 128},  # 实验1
            {'learning_rate': 0.001, 'hidden_size': 256, 'embedding_dim': 100, 'kernel_sizes': [3, 4, 5], 'num_filters': 128},  # 实验2
            {'learning_rate': 0.0001, 'hidden_size': 128, 'embedding_dim': 100, 'kernel_sizes': [3, 4, 5], 'num_filters': 128}, # 实验3
            {'learning_rate': 0.001, 'hidden_size': 128, 'embedding_dim': 100, 'kernel_sizes': [2, 3, 4], 'num_filters': 256},  # 实验4
            {'learning_rate': 0.0005, 'hidden_size': 256, 'embedding_dim': 100, 'kernel_sizes': [3, 4, 5], 'num_filters': 256}   # 实验5
        ]
        
        # 加载数据
        preprocessor = TextPreprocessor()
        train_reviews, train_labels = preprocessor.load_data('train.csv')
        test_reviews, test_labels = preprocessor.load_data('test.csv')
        valid_reviews, valid_labels = preprocessor.load_data('valid.csv')
        
        # 分词处理
        print("CNN实验：分词处理中...")
        train_seg = preprocessor.tokenize(train_reviews)
        test_seg = preprocessor.tokenize(test_reviews)
        valid_seg = preprocessor.tokenize(valid_reviews)
        
        # 构建词汇表
        preprocessor.build_vocab(train_seg, min_freq=3)
        vocab_size = len(preprocessor.word_index)
        
        # 转换数据为序列
        max_len = 100
        train_seqs = preprocessor.texts_to_sequences(train_seg)
        train_seqs = preprocessor.pad_sequences(train_seqs)
        test_seqs = preprocessor.texts_to_sequences(test_seg)
        test_seqs = preprocessor.pad_sequences(test_seqs)
        valid_seqs = preprocessor.texts_to_sequences(valid_seg)
        valid_seqs = preprocessor.pad_sequences(valid_seqs)
        
        for i in range(min(num_experiments, len(param_combinations))):
            print(f"\nCNN实验 {i+1}/{num_experiments}")
            params = param_combinations[i]
            
            # 初始化模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CNNTextClassifier(
                vocab_size=vocab_size,
                embedding_dim=params['embedding_dim'],
                num_filters=params['num_filters'],
                filter_sizes=params['kernel_sizes']
            ).to(device)
            
            # 训练参数
            batch_size = 64
            num_epochs = 5  # 减少训练轮数以节省时间
            
            # 创建数据加载器
            train_dataset = TextDataset(train_seqs, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = TextDataset(test_seqs, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            valid_dataset = TextDataset(valid_seqs, valid_labels)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
            
            # 训练模型
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            start_time = time.time()
            total_loss = 0.0
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # 调整标签形状为[batch_size, 1]
                    labels = labels.view(-1, 1)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * inputs.size(0)
                
                epoch_loss = epoch_loss / len(train_loader.dataset)
                total_loss += epoch_loss
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            train_time = time.time() - start_time
            avg_train_loss = total_loss / num_epochs
            
            # 测试集准确率
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float().squeeze()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            test_accuracy = correct / total
            
            # 验证集预测耗时
            start_time = time.time()
            correct_valid = 0
            total_valid = 0
            
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float().squeeze()
                    total_valid += labels.size(0)
                    correct_valid += (predicted == labels).sum().item()
            
            total_valid_time = time.time() - start_time
            valid_time_per_100 = (total_valid_time / total_valid) * 100
            
            # 记录结果
            self.add_result(
                model_type='CNN',
                exp_num=i+1,
                learning_rate=params['learning_rate'],
                hidden_size=params['hidden_size'],
                train_loss=avg_train_loss,
                test_accuracy=test_accuracy,
                valid_time_per_100=valid_time_per_100,
                train_time=train_time
            )
            
            print(f"实验 {i+1} 完成: 学习率={params['learning_rate']}, 隐藏层={params['hidden_size']}, 测试准确率={test_accuracy:.4f}")
    
    def run_lstm_experiments(self, num_experiments=5):
        """运行LSTM模型实验"""
        from lstm_text_classifier import LSTMClassifier, TextDataset, LSTMTextClassifier
        
        # 定义超参数组合
        param_combinations = [
            {'learning_rate': 0.001, 'hidden_size': 128, 'embedding_dim': 100, 'num_layers': 1},  # 实验1
            {'learning_rate': 0.001, 'hidden_size': 256, 'embedding_dim': 100, 'num_layers': 1},  # 实验2
            {'learning_rate': 0.0001, 'hidden_size': 128, 'embedding_dim': 100, 'num_layers': 1}, # 实验3
            {'learning_rate': 0.001, 'hidden_size': 128, 'embedding_dim': 100, 'num_layers': 2},  # 实验4
            {'learning_rate': 0.0005, 'hidden_size': 256, 'embedding_dim': 100, 'num_layers': 2}   # 实验5
        ]
        
        # 加载数据
        preprocessor = TextPreprocessor()
        train_reviews, train_labels = preprocessor.load_data('train.csv')
        test_reviews, test_labels = preprocessor.load_data('test.csv')
        valid_reviews, valid_labels = preprocessor.load_data('valid.csv')
        
        # 分词处理
        print("LSTM实验：分词处理中...")
        train_seg = preprocessor.tokenize(train_reviews)
        test_seg = preprocessor.tokenize(test_reviews)
        valid_seg = preprocessor.tokenize(valid_reviews)
        
        # 构建词汇表
        preprocessor.build_vocab(train_seg, min_freq=3)
        vocab_size = len(preprocessor.word_index)
        
        # 转换数据为序列
        max_len = 100
        train_seqs = preprocessor.texts_to_sequences(train_seg)
        train_seqs = preprocessor.pad_sequences(train_seqs)
        test_seqs = preprocessor.texts_to_sequences(test_seg)
        test_seqs = preprocessor.pad_sequences(test_seqs)
        valid_seqs = preprocessor.texts_to_sequences(valid_seg)
        valid_seqs = preprocessor.pad_sequences(valid_seqs)
        
        for i in range(min(num_experiments, len(param_combinations))):
            print(f"\nLSTM实验 {i+1}/{num_experiments}")
            params = param_combinations[i]
            
            # 初始化模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LSTMTextClassifier(
                vocab_size=vocab_size,
                embedding_dim=params['embedding_dim'],
                hidden_dim=params['hidden_size'],
                num_layers=params['num_layers']
            ).to(device)
            
            # 训练参数
            batch_size = 64
            num_epochs = 5  # 减少训练轮数以节省时间
            
            # 创建数据加载器
            train_dataset = TextDataset(train_seqs, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = TextDataset(test_seqs, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            valid_dataset = TextDataset(valid_seqs, valid_labels)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
            
            # 训练模型
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            start_time = time.time()
            total_loss = 0.0
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # 调整标签形状为[batch_size, 1]
                    labels = labels.view(-1, 1)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * inputs.size(0)
                
                epoch_loss = epoch_loss / len(train_loader.dataset)
                total_loss += epoch_loss
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            train_time = time.time() - start_time
            avg_train_loss = total_loss / num_epochs
            
            # 测试集准确率
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float().squeeze()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            test_accuracy = correct / total
            
            # 验证集预测耗时
            start_time = time.time()
            correct_valid = 0
            total_valid = 0
            
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float().squeeze()
                    total_valid += labels.size(0)
                    correct_valid += (predicted == labels).sum().item()
            
            total_valid_time = time.time() - start_time
            valid_time_per_100 = (total_valid_time / total_valid) * 100
            
            # 记录结果
            self.add_result(
                model_type='LSTM',
                exp_num=i+1,
                learning_rate=params['learning_rate'],
                hidden_size=params['hidden_size'],
                train_loss=avg_train_loss,
                test_accuracy=test_accuracy,
                valid_time_per_100=valid_time_per_100,
                train_time=train_time
            )
            
            print(f"实验 {i+1} 完成: 学习率={params['learning_rate']}, 隐藏层={params['hidden_size']}, 层数={params['num_layers']}, 测试准确率={test_accuracy:.4f}")

def main():
    # 创建实验管理器
    manager = ExperimentManager('experiment_results.csv')
    
    # 运行各模型实验
    print("=" * 50)
    print("开始运行贝叶斯模型实验")
    print("=" * 50)
    manager.run_bayes_experiments(num_experiments=3)
    
    print("\n" + "=" * 50)
    print("开始运行SVM模型实验")
    print("=" * 50)
    manager.run_svm_experiments(num_experiments=5)
    
    print("\n" + "=" * 50)
    print("开始运行CNN模型实验")
    print("=" * 50)
    manager.run_cnn_experiments(num_experiments=5)
    
    print("\n" + "=" * 50)
    print("开始运行LSTM模型实验")
    print("=" * 50)
    manager.run_lstm_experiments(num_experiments=5)
    
    print("\n" + "=" * 50)
    print("所有实验完成！结果已保存到 experiment_results.csv")
    print("=" * 50)

if __name__ == "__main__":
    main()
