import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 加载训练好的词向量模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# 加载句子并进行分词
def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return list(sentences)

# 将句子转为向量（平均词向量）
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        valid_words = 0
        for word in words:
            if word in model.wv:
                vector += model.wv[word]
                valid_words += 1
        if valid_words > 0:
            vector /= valid_words
        vectors.append(vector)
    return np.array(vectors)

def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")    # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 自动计算聚类数
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)

    # 计算每个样本到聚类中心的距离
    distances = kmeans.transform(vectors)  # shape: (n_samples, n_clusters)

    # 构建字典：label -> [(sentence, distance)]
    sentence_label_dict = defaultdict(list)
    for sentence, label, dist_vec in zip(sentences, kmeans.labels_, distances):
        distance_to_center = dist_vec[label]  # 当前样本到自己类中心的距离
        sentence_label_dict[label].append((sentence, distance_to_center))

    # 对每个类内句子按距离升序排列（越小越接近中心）
    for label, sentence_dist_list in sentence_label_dict.items():
        sentence_dist_list.sort(key=lambda x: x[1])

    # 输出结果
    for label, sentence_dist_list in sentence_label_dict.items():
        print(f"\nCluster {label} (样本数: {len(sentence_dist_list)})")
        print("Top 10 代表性句子（按距离中心由近到远排序）:")
        for i, (sentence, dist) in enumerate(sentence_dist_list[:10]):
            print(f"{i+1}. {sentence.replace(' ', '')}  [距离={dist:.4f}]")
        print("---------")

if __name__ == "__main__":
    main()
