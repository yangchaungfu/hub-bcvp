#!/usr/bin/env python3  
#coding: utf-8
import os
# 设置环境变量 OMP_NUM_THREADS=8（仅当前脚本生效）
os.environ["OMP_NUM_THREADS"] = "8"

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
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

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

#进行距离计算
def word_distance(sentence1, sentence2):
    distance = 0
    for i in range(len(sentence1)):
        # pow是内置函数，核心作用是 计算幂运算（即 x 的 y 次方）  x ** y
        # distance = sum(pow(s1 - s2, 2) for s1,s2 in zip(sentence1, sentence2))
    # return pow(distance, 0.5)
        # 使用欧式距离
        distance = sum((s1 - s2) ** 2 for s1, s2 in zip(sentence1, sentence2))
    return math.sqrt(distance)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    #在 Python 中，collections.defaultdict 是内置 dict（字典）的子类，
    # 核心作用是：为字典中「不存在的键」提供一个默认值（无需手动判断键是否存在），
    # 避免了普通字典访问不存在的键时抛出 KeyError 异常
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        #kmeans.cluester_centers_ #每个聚类中心 
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    label_distance_dict = defaultdict(float)
    for label, sentence_tmp in sentence_label_dict.items():
        label_vectors = sentences_to_vectors(sentence_tmp, model) # 每个聚类向量化
        label_center_distance = kmeans.cluster_centers_[label] # 每个聚类中心的距离
        distance = 0
        for label_ver in label_vectors:
            distance += word_distance(label_ver,label_center_distance)
        label_distance_dict[label] = distance/len(sentence_tmp)
    # print(label_vectors)
    # print("=========================")
    # 2. 按距离从小到大排序（sorted 默认升序）
    # items() 提取 (key, value) 元组，key=lambda x: x[0] 表示按元组第 0 位（键）排序
    # list排序后的键值对元组（label, distance）
    sorted_label_distance_list = sorted(label_distance_dict.items(), key=lambda x: (x[1], x[0]))
    # print(sorted_label_distance_list)
    for label, distance in sorted_label_distance_list:
        sentences = sentence_label_dict[label]
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

