import time
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====================== 1. 配置参数 ======================
# 停用词（基础版，可自行补充）
STOPWORDS = ['的', '了', '是', '我', '你', '他', '她', '它', '在', '和', '有', '就', '都', '不', '也', '这', '那']
# 数据路径（替换为你的CSV路径）
DATA_PATH = "ecommerce_comments.csv"
# 模型参数
MAX_SEQ_LEN = 50  # LSTM序列最大长度
EMBEDDING_DIM = 128  # 词嵌入维度
BATCH_SIZE = 32
EPOCHS = 5
TEST_SIZE = 0.2  # 验证集比例

# ====================== 2. 数据读取与预处理 ======================
def preprocess_text(text):
    """文本预处理：去空格、分词、去停用词"""
    if pd.isna(text):
        return ""
    # 去空格
    text = text.strip()
    # 分词
    words = jieba.lcut(text)
    # 去停用词和空字符
    words = [w for w in words if w not in STOPWORDS and w != ""]
    return " ".join(words)

# 读取数据
df = pd.read_csv("文本分类练习.csv")
# 基础清洗：删除空值
df = df.dropna(subset=["review", "label"])
# 文本预处理
df["processed_text"] = df["review"].apply(preprocess_text)

# 划分训练集/验证集
X = df["processed_text"].values
y = df["label"].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

# ====================== 3. 数据分析 ======================
print("="*50 + " 数据分析结果 " + "="*50)
# 3.1 正负样本统计
pos_num = sum(y == 1)
neg_num = sum(y == 0)
total_num = len(y)
print(f"总样本数：{total_num}")
print(f"好评样本数：{pos_num} ({pos_num/total_num:.2%})")
print(f"差评样本数：{neg_num} ({neg_num/total_num:.2%})")

# 3.2 文本长度分析（分词后的词数）
df["text_len"] = df["processed_text"].apply(lambda x: len(x.split()))
avg_len = df["text_len"].mean()
max_len = df["text_len"].max()
min_len = df["text_len"].min()
median_len = df["text_len"].median()
print(f"文本长度统计：")
print(f"  平均长度：{avg_len:.2f} 词")
print(f"  最大长度：{max_len} 词")
print(f"  最小长度：{min_len} 词")
print(f"  中位数长度：{median_len} 词")


# ====================== 4. 特征工程（为Bayes/SVM准备） ======================
# TF-IDF向量化（文本转数值特征）
tfidf = TfidfVectorizer(max_features=5000)  # 限制最大特征数，避免维度爆炸
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# ====================== 5. 模型训练与评估 ======================
# 存储结果的字典
results = {
    "model": [],
    "accuracy": [],
    "f1_score": [],
    "predict_time_ms_per_sample": []
}

# 5.1 朴素贝叶斯（Bayes）
print("\n" + "="*50 + " 训练朴素贝叶斯模型 " + "="*50)
bayes_model = MultinomialNB()
bayes_model.fit(X_train_tfidf, y_train)

# 预测速度测试（批量预测1000个样本，取平均）
start_time = time.time()
bayes_pred = bayes_model.predict(X_val_tfidf[:1000])
end_time = time.time()
bayes_time_per_sample = (end_time - start_time) * 1000 / 1000  # 毫秒/样本

# 评估
bayes_acc = accuracy_score(y_val, bayes_model.predict(X_val_tfidf))
bayes_f1 = f1_score(y_val, bayes_model.predict(X_val_tfidf))
print(f"朴素贝叶斯 - 准确率：{bayes_acc:.4f}，F1-score：{bayes_f1:.4f}")

# 记录结果
results["model"].append("朴素贝叶斯")
results["accuracy"].append(round(bayes_acc, 4))
results["f1_score"].append(round(bayes_f1, 4))
results["predict_time_ms_per_sample"].append(round(bayes_time_per_sample, 4))

# 5.2 SVM
print("\n" + "="*50 + " 训练SVM模型 " + "="*50)
svm_model = SVC(kernel="linear", probability=True, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# 预测速度测试
start_time = time.time()
svm_pred = svm_model.predict(X_val_tfidf[:1000])
end_time = time.time()
svm_time_per_sample = (end_time - start_time) * 1000 / 1000

# 评估
svm_acc = accuracy_score(y_val, svm_model.predict(X_val_tfidf))
svm_f1 = f1_score(y_val, svm_model.predict(X_val_tfidf))
print(f"SVM - 准确率：{svm_acc:.4f}，F1-score：{svm_f1:.4f}")

# 记录结果
results["model"].append("SVM")
results["accuracy"].append(round(svm_acc, 4))
results["f1_score"].append(round(svm_f1, 4))
results["predict_time_ms_per_sample"].append(round(svm_time_per_sample, 4))

# 5.3 LSTM（需重新处理文本为序列特征）
print("\n" + "="*50 + " 训练LSTM模型 " + "="*50)
# 词表构建
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
# 文本转序列
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
# 序列填充/截断（统一长度）
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LEN)
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_SEQ_LEN)

# 构建LSTM模型
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# 训练
lstm_model.fit(X_train_pad, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val_pad, y_val))

# 预测速度测试
start_time = time.time()
lstm_pred = (lstm_model.predict(X_val_pad[:1000], verbose=0) > 0.5).astype(int).flatten()
end_time = time.time()
lstm_time_per_sample = (end_time - start_time) * 1000 / 1000

# 评估
lstm_acc = accuracy_score(y_val, (lstm_model.predict(X_val_pad, verbose=0) > 0.5).astype(int).flatten())
lstm_f1 = f1_score(y_val, (lstm_model.predict(X_val_pad, verbose=0) > 0.5).astype(int).flatten())
print(f"LSTM - 准确率：{lstm_acc:.4f}，F1-score：{lstm_f1:.4f}")

# 记录结果
results["model"].append("LSTM")
results["accuracy"].append(round(lstm_acc, 4))
results["f1_score"].append(round(lstm_f1, 4))
results["predict_time_ms_per_sample"].append(round(lstm_time_per_sample, 4))

# 6. 结果表格输出
print("\n" + "="*80)
print("模型对比结果汇总")
print("-"*80)
# 打印表头（对齐格式）
print(f"{'模型':<10} {'准确率':<10} {'F1-score':<10} {'单样本预测时间(ms)':<15}")
print("-"*80)
# 遍历结果字典，逐行打印数据（对齐格式）
for i in range(len(results["model"])):
    model = results["model"][i]
    acc = results["accuracy"][i]
    f1 = results["f1_score"][i]
    time_per_sample = results["predict_time_ms_per_sample"][i]
    # 格式化输出，保证列对齐
    print(f"{model:<10} {acc:<10.4f} {f1:<10.4f} {time_per_sample:<15.4f}")
print("-"*80)
