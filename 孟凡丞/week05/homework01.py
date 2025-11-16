import numpy as np
import random


class MeansClusterer:
    def __init__(self, data, k, max_iter=100, random_state=None):
        """
        初始化聚类器
        :param data: 输入数据，形状为 (n_samples, n_features)，类型为 numpy.ndarray
        :param k: 要聚成的簇数量
        :param max_iter: 最大迭代次数，防止不收敛时死循环
        :param random_state: 随机种子
        """
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        # NumPy数组
        self.data = np.array(data)
        self.k = k
        self.max_iter = max_iter

        n_samples, n_features = self.data.shape
        if k <= 0 or k > n_samples:
            raise ValueError(f"簇数 k 必须满足 1 <= k <= {n_samples}，当前 k={k}")

        # 随机选择 k 个不同的样本作为初始质心
        indices = np.random.choice(n_samples, size=k, replace=False)
        # 初始质心
        self.centers = self.data[indices].copy()

    def _euclidean_distance(self, point, center):
        """计算两点间的欧氏距离"""
        return np.linalg.norm(point - center)

    def _assign_clusters(self):
        """将每个点分配给最近的质心，返回每个点所属的簇索引"""
        assignments = []
        for point in self.data:
            # 计算该点到所有质心的距离
            distances = [self._euclidean_distance(point, center) for center in self.centers]
            # 找到最近的质心索引
            closest_cluster = np.argmin(distances)
            assignments.append(closest_cluster)
        return np.array(assignments)

    def _update_centers(self, assignments):
        """根据当前分配结果，更新每个簇的质心"""
        new_centers = np.zeros_like(self.centers)
        for i in range(self.k):
            # 获取属于第 i 个簇的所有点
            cluster_points = self.data[assignments == i]

            # 处理空簇：如果某个簇没有点，则重新随机选一个点作为新质心
            if len(cluster_points) == 0:
                print(f"警告：簇 {i} 为空，重新随机初始化其质心。")
                random_idx = np.random.randint(0, len(self.data))
                new_centers[i] = self.data[random_idx]
            else:
                # 否则取均值作为新质心
                new_centers[i] = cluster_points.mean(axis=0)
        return new_centers

    def _compute_intra_cluster_distances(self, assignments):
        """
        计算每个簇的类内总距离（即簇内所有点到质心的距离之和）
        返回长度为 k 的列表，每个元素对应一个簇的总距离
        """
        intra_distances = []
        for i in range(self.k):
            cluster_points = self.data[assignments == i]
            if len(cluster_points) == 0:
                intra_distances.append(0.0)  # 空簇距离为0
            else:
                total_dist = sum(self._euclidean_distance(p, self.centers[i]) for p in cluster_points)
                intra_distances.append(total_dist)
        return intra_distances

    def fit(self):
        """
        执行 K-means 聚类主循环
        返回：
          clusters: list of lists，每个子列表包含属于该簇的原始数据点（list 形式）
          centers: 最终质心（numpy array）
          intra_distances: 每个簇的类内总距离（list）
        """
        for iteration in range(self.max_iter):
            # 步骤1：分配每个点到最近的簇
            assignments = self._assign_clusters()

            # 步骤2：更新质心
            new_centers = self._update_centers(assignments)

            # 步骤3：检查是否收敛（质心变化很小）
            if np.allclose(self.centers, new_centers, atol=1e-6):
                print(f"K-means 在第 {iteration + 1} 次迭代后收敛。")
                break

            self.centers = new_centers
        else:
            print(f"达到最大迭代次数 {self.max_iter}，未完全收敛。")

        # 最终分配
        final_assignments = self._assign_clusters()

        # 构建每个簇的点列表
        clusters = []
        for i in range(self.k):
            cluster_points = self.data[final_assignments == i]
            clusters.append(cluster_points.tolist())  # 转为 Python list 便于打印

        # 计算每个簇的类内距离
        intra_distances = self._compute_intra_cluster_distances(final_assignments)

        return clusters, self.centers, intra_distances


def sort_clusters_by_intra_distance(clusters, centers, intra_distances):
    """
    根据类内距离对簇进行排序（从小到大：最紧凑 → 最松散）
    返回排序后的列表，每个元素为 (簇索引, 类内距离, 簇点列表, 质心)
    """
    # 创建带索引的元组列表
    indexed_clusters = [
        (i, intra_distances[i], clusters[i], centers[i])
        for i in range(len(clusters))
    ]
    # 按类内距离升序排序
    sorted_clusters = sorted(indexed_clusters, key=lambda x: x[1])
    return sorted_clusters


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 生成合理的随机数据：100 个样本，4 维特征
    np.random.seed(42)  # 保证每次运行结果一致
    data = np.random.rand(100, 4) * 10  # 缩放到 [0, 10) 区间

    # 初始化并运行 K-means（k=5）
    kmeans = MeansClusterer(data=data, k=5, max_iter=100, random_state=123)
    clusters, centers, intra_distances = kmeans.fit()

    # 按类内距离排序
    sorted_result = sort_clusters_by_intra_distance(clusters, centers, intra_distances)

    # 打印排序结果
    print("\n=== 按类内距离排序的聚类结果（从小到大） ===")
    for rank, (orig_idx, dist, points, center) in enumerate(sorted_result, start=1):
        print(f"\n【第 {rank} 紧凑的簇】")
        print(f"  原始簇索引: {orig_idx}")
        print(f"  类内总距离: {dist:.4f}")
        print(f"  质心坐标: {center}")
        print(f"  包含点数: {len(points)}")