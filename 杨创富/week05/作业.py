import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict

class TextKMeansSorter:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.vectorizer = None
        self.cluster_centers = None
        
    def preprocess_texts(self, texts):
        """文本向量化处理"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(texts)
        return X
    
    def fit_cluster(self, texts):
        """执行KMeans聚类"""
        # 文本向量化
        X = self.preprocess_texts(texts)
        
        # KMeans聚类
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        labels = self.kmeans.fit_predict(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        return labels, X
    
    def calculate_intra_cluster_distances(self, texts, labels, X):
        """计算类内距离并排序"""
        
        # 将文本、标签和特征向量组合
        results = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            # 计算当前文本到其簇质心的距离（使用余弦距离）
            text_vector = X[i].toarray()
            centroid = self.cluster_centers[label].reshape(1, -1)
            
            # 使用余弦相似度，然后转换为距离
            similarity = cosine_similarity(text_vector, centroid)[0][0]
            distance = 1 - similarity  # 余弦距离
            
            results.append({
                'text': text,
                'cluster': label,
                'distance_to_centroid': distance,
                'vector': text_vector
            })
        
        return results
    
    def sort_clusters_by_intra_distance(self, results):
        """根据类内距离对簇进行排序"""
        
        # 按簇分组
        cluster_groups = defaultdict(list)
        for item in results:
            cluster_groups[item['cluster']].append(item)
        
        # 计算每个簇的平均类内距离
        cluster_stats = []
        for cluster_id, items in cluster_groups.items():
            avg_distance = np.mean([item['distance_to_centroid'] for item in items])
            cluster_stats.append({
                'cluster_id': cluster_id,
                'avg_intra_distance': avg_distance,
                'cluster_size': len(items)
            })
        
        # 按平均类内距离排序（距离越大，簇越分散）
        sorted_clusters = sorted(cluster_stats, 
                               key=lambda x: x['avg_intra_distance'], 
                               reverse=True)
        
        return sorted_clusters, cluster_groups
    
    def sort_texts_within_clusters(self, cluster_groups):
        """对每个簇内的文本按到质心距离排序"""
        sorted_results = {}
        
        for cluster_id, items in cluster_groups.items():
            # 按到质心距离排序（距离越小，越接近中心）
            sorted_items = sorted(items, 
                                key=lambda x: x['distance_to_centroid'])
            sorted_results[cluster_id] = sorted_items
            
        return sorted_results
    
    def process(self, texts):
        """完整的处理流程"""
        print("步骤1: 执行KMeans聚类...")
        labels, X = self.fit_cluster(texts)
        
        print("步骤2: 计算类内距离...")
        results = self.calculate_intra_cluster_distances(texts, labels, X)
        
        print("步骤3: 对簇按类内距离排序...")
        sorted_clusters, cluster_groups = self.sort_clusters_by_intra_distance(results)
        
        print("步骤4: 对每个簇内的文本排序...")
        sorted_texts = self.sort_texts_within_clusters(cluster_groups)
        
        return {
            'sorted_clusters': sorted_clusters,
            'sorted_texts': sorted_texts,
            'labels': labels
        }

# 示例使用
def demo():
    # 示例文本数据
    sample_texts = [
        "machine learning algorithms and deep neural networks",
        "artificial intelligence and computer science",
        "data mining techniques and big data analytics",
        "natural language processing and text analysis",
        "computer vision and image recognition",
        "reinforcement learning and robotics",
        "statistical analysis and probability theory",
        "python programming and software development",
        "java programming and web development",
        "cloud computing and distributed systems",
        "database management and sql queries",
        "network security and cybersecurity",
        "operating systems and computer architecture",
        "mobile app development and ios android",
        "web design and user interface development",
        "business intelligence and data visualization",
        "project management and agile methodology",
        "customer service and support techniques",
        "digital marketing and social media strategy",
        "financial analysis and investment strategies"
    ]
    
    # 创建排序器实例
    sorter = TextKMeansSorter(n_clusters=4)
    
    # 执行处理
    results = sorter.process(sample_texts)
    
    # 输出结果
    print("\n" + "="*50)
    print("聚类排序结果（按类内距离从大到小）：")
    print("="*50)
    
    for i, cluster_info in enumerate(results['sorted_clusters']):
        cluster_id = cluster_info['cluster_id']
        avg_distance = cluster_info['avg_intra_distance']
        size = cluster_info['cluster_size']
        
        print(f"\n簇 {cluster_id} (平均类内距离: {avg_distance:.4f}, 包含{size}个文本):")
        print("-" * 40)
        
        # 输出该簇内排序后的文本
        sorted_texts_in_cluster = results['sorted_texts'][cluster_id]
        for j, item in enumerate(sorted_texts_in_cluster):
            print(f"  {j+1}. [距离: {item['distance_to_centroid']:.4f}] {item['text']}")

# 进阶功能：可视化类内距离分布
def visualize_cluster_distances(results):
    """可视化类内距离分布"""
    import matplotlib.pyplot as plt
    
    clusters_info = results['sorted_clusters']
    cluster_ids = [f'Cluster {info["cluster_id"]}' for info in clusters_info]
    avg_distances = [info['avg_intra_distance'] for info in clusters_info]
    sizes = [info['cluster_size'] for info in clusters_info]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 平均类内距离条形图
    bars = ax1.bar(cluster_ids, avg_distances, color='skyblue')
    ax1.set_title('各簇平均类内距离')
    ax1.set_ylabel('平均距离')
    ax1.set_xlabel('簇')
    
    # 在条形上添加数值
    for bar, distance in zip(bars, avg_distances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{distance:.4f}', ha='center', va='bottom')
    
    # 簇大小饼图
    ax2.pie(sizes, labels=cluster_ids, autopct='%1.1f%%', startangle=90)
    ax2.set_title('各簇文本数量分布')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
    
    # 如果需要可视化，取消下面的注释
    # sample_texts = [...] # 你的文本数据
    # sorter = TextKMeansSorter(n_clusters=4)
    # results = sorter.process(sample_texts)
    # visualize_cluster_distances(results)
