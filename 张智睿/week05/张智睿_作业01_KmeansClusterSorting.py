import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


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
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算每个聚类的类内平均距离
def calculate_intra_cluster_distances(vectors, labels, cluster_centers):
    intra_distances = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_vectors = vectors[labels == label]

        if len(cluster_vectors) == 0:
            intra_distances[label] = 0
            continue

        center = cluster_centers[label]
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        intra_distances[label] = np.mean(distances)

    return intra_distances


def main():
    model = load_word2vec_model(r"model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("=" * 50)
    print("句子总数：", len(sentences))
    print("指定聚类数量：", n_clusters)

    kmeans = KMeans(n_clusters)  # 移除了random_state参数
    kmeans.fit(vectors)

    # 计算类内平均距离
    intra_distances = calculate_intra_cluster_distances(vectors, kmeans.labels_, kmeans.cluster_centers_)

    # 验证聚类结果
    print("实际聚类数量：", kmeans.n_clusters)
    print("唯一标签数量：", len(set(kmeans.labels_)))
    print("=" * 50)

    # 组织聚类结果
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    # 按类内平均距离从小到大排序
    sorted_clusters = sorted(
        [(label, cluster_sentences, intra_distances[label])
         for label, cluster_sentences in sentence_label_dict.items()],
        key=lambda x: x[2]  # 按类内平均距离排序
    )

    # 将结果保存到文件
    with open("clustering_results.txt", "w", encoding="utf-8") as f:
        f.write("=== 文本聚类分析结果 ===\n")
        f.write(f"总句子数: {len(sentences)}\n")
        f.write(f"聚类数量: {n_clusters}\n")
        f.write("排序方式: 按类内平均距离从小到大\n\n")

        for new_idx, (old_label, cluster_sentences, avg_distance) in enumerate(sorted_clusters):
            f.write(
                f"聚类 {new_idx} (原标签{old_label}, 类内平均距离: {avg_distance:.4f}, 包含{len(cluster_sentences)}个句子):\n")
            for i in range(min(10, len(cluster_sentences))):
                f.write(f"  {i + 1}. {cluster_sentences[i].replace(' ', '')}\n")
            f.write("\n")

    # 控制台输出
    print("\n=== 聚类结果预览 (按类内平均距离排序) ===")
    for new_idx, (old_label, cluster_sentences, avg_distance) in enumerate(sorted_clusters):
        print("聚类 {} (原标签{}, 类内平均距离: {:.4f}, 包含{}个句子):".format(
            new_idx, old_label, avg_distance, len(cluster_sentences)))
        for i in range(min(5, len(cluster_sentences))):
            print("  " + cluster_sentences[i].replace(" ", ""))
        print("---------")

    print("\n详细结果已保存到: clustering_results.txt")

    # 输出平均类内距离
    all_distances = [distance for _, _, distance in sorted_clusters]
    print(f"\n平均类内距离: {np.mean(all_distances):.4f}")


if __name__ == "__main__":
    main()
