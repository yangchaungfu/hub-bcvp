import pandas as pd
import numpy as np
import jieba
from collections import Counter

class TextPreprocessor:
    def __init__(self, max_words=10000, max_length=100):
        self.max_words = max_words  # 词汇表大小
        self.max_length = max_length  # 文本最大长度
        self.word_index = {}  # 词到索引的映射
        self.index_word = {}  # 索引到词的映射
        self.vocab_size = 0  # 实际词汇表大小

    def load_data(self, csv_path):
        """从CSV文件加载数据"""
        df = pd.read_csv(csv_path)
        labels = df['label'].values
        reviews = df['review'].values
        return reviews, labels

    def tokenize(self, reviews):
        """使用jieba分词处理文本"""
        tokenized_reviews = []
        for review in reviews:
            tokens = jieba.lcut(review)
            tokenized_reviews.append(tokens)
        return tokenized_reviews

    def build_vocab(self, tokenized_reviews, min_freq=1):
        """构建词汇表"""
        # 统计词频
        word_counts = Counter()
        for tokens in tokenized_reviews:
            word_counts.update(tokens)
        
        # 过滤掉词频低于min_freq的词
        filtered_words = [(word, count) for word, count in word_counts.items() if count >= min_freq]
        
        # 按词频排序，选择前max_words个词
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)[:self.max_words]
        
        # 构建词到索引的映射，0保留给padding，1保留给未知词
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(sorted_words, 2):
            self.word_index[word] = i
        
        # 构建索引到词的映射
        self.index_word = {i: word for word, i in self.word_index.items()}
        
        self.vocab_size = len(self.word_index)
        print(f"词汇表大小: {self.vocab_size}")
        return

    def texts_to_sequences(self, tokenized_reviews):
        """将文本转换为索引序列"""
        sequences = []
        for tokens in tokenized_reviews:
            sequence = [self.word_index.get(token, 1) for token in tokens]  # 未知词用1表示
            sequences.append(sequence)
        return sequences

    def pad_sequences(self, sequences):
        """对序列进行padding，使其长度一致"""
        padded_sequences = np.zeros((len(sequences), self.max_length), dtype=np.int32)
        for i, sequence in enumerate(sequences):
            length = min(len(sequence), self.max_length)
            padded_sequences[i, :length] = sequence[:length]
        return padded_sequences

    def preprocess_data(self, csv_path, fit_vocab=True):
        """完整的数据预处理流程"""
        # 加载数据
        reviews, labels = self.load_data(csv_path)
        
        # 分词
        tokenized_reviews = self.tokenize(reviews)
        
        # 构建词汇表（只在训练集上构建）
        if fit_vocab:
            self.build_vocab(tokenized_reviews)
        
        # 转换为序列
        sequences = self.texts_to_sequences(tokenized_reviews)
        
        # Padding
        padded_sequences = self.pad_sequences(sequences)
        
        return padded_sequences, labels

    def save_vocab(self, vocab_path):
        """保存词汇表"""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word, index in self.word_index.items():
                f.write(f"{word},{index}\n")

    def load_vocab(self, vocab_path):
        """加载词汇表"""
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, index = line.strip().split(',')
                self.word_index[word] = int(index)
        self.index_word = {i: word for word, i in self.word_index.items()}
        self.vocab_size = len(self.word_index)
        print(f"已加载词汇表，大小: {self.vocab_size}")
        return