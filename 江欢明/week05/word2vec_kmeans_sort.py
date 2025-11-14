#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算每个聚类的内部距离
    cluster_distances = []
    for label in range(n_clusters):
        # 获取当前聚类中的所有向量
        cluster_vectors = vectors[kmeans.labels_ == label]

        if len(cluster_vectors) > 0:
            # 计算聚类内所有点到聚类中心的平均距离
            center = kmeans.cluster_centers_[label]
            distances = np.linalg.norm(cluster_vectors - center, axis=1)
            avg_distance = np.mean(distances)
            cluster_distances.append((label, avg_distance, len(cluster_vectors)))

    # 按聚类内距离排序（从小到大）
    cluster_distances.sort(key=lambda x: x[1])

    # 组织句子到对应的聚类标签
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    # 只打印聚类内距离最小的10个类
    print("\n聚类内距离最小的10个类：")
    for i, (label, avg_distance, count) in enumerate(cluster_distances[:10]):
        print(f"\n聚类 {label} (平均距离: {avg_distance:.4f}, 样本数: {count}):")
        for j in range(min(10, len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][j].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

