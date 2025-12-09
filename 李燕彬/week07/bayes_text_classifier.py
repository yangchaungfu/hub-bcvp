import math
import jieba
import re
import os
import json
from collections import defaultdict
import pandas as pd

jieba.initialize()

class BayesTextClassifier:
    def __init__(self):
        self.p_class = defaultdict(int)  # 类别先验概率
        self.word_class_prob = defaultdict(dict)  # 词在类别中的条件概率
        self.class_name_to_word_freq = defaultdict(dict)  # 类别到词频的映射
        self.all_words = set()  # 词表

    def load_data_from_csv(self, csv_path):
        """从CSV文件加载数据"""
        df = pd.read_csv(csv_path)
        return df

    def train(self, train_data):
        """训练贝叶斯分类器"""
        # 重置模型参数
        self.p_class = defaultdict(int)
        self.class_name_to_word_freq = defaultdict(dict)
        self.all_words = set()
        
        # 遍历训练数据
        for _, row in train_data.iterrows():
            label = str(row['label'])  # 将label转换为字符串作为类别名
            review = row['review']
            
            # 使用jieba分词
            words = jieba.lcut(review)
            
            # 更新词表
            self.all_words = self.all_words.union(set(words))
            
            # 更新类别计数
            self.p_class[label] += 1
            
            # 更新类别下的词频
            word_freq = self.class_name_to_word_freq[label]
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        
        # 将词频转换为概率
        self._freq_to_prob()
        print(f"训练完成！共训练 {len(train_data)} 个样本，识别到 {len(self.all_words)} 个不同的词")
        return

    def _freq_to_prob(self):
        """将词频和样本频率转换为概率"""
        # 计算先验概率 P(x)
        total_sample_count = sum(self.p_class.values())
        self.p_class = dict([(c, self.p_class[c] / total_sample_count) for c in self.p_class])
        
        # 计算条件概率 P(w|x)
        self.word_class_prob = defaultdict(dict)
        for class_name, word_freq in self.class_name_to_word_freq.items():
            total_word_count = sum(count for count in word_freq.values())  # 类别总词数
            for word in word_freq:
                # 加1平滑，避免出现概率为0
                prob = (word_freq[word] + 1) / (total_word_count + len(self.all_words))
                self.word_class_prob[class_name][word] = prob
            # 未知词的概率
            self.word_class_prob[class_name]["<unk>"] = 1 / (total_word_count + len(self.all_words))
        return

    def predict(self, review):
        """预测单个文本的类别"""
        words = jieba.lcut(review)
        results = []
        
        # 计算每个类别的概率
        for class_name in self.p_class:
            prob = self._get_class_prob(words, class_name)
            results.append([class_name, prob])
        
        # 按概率降序排序
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # 计算公共分母，将结果转换为0-1之间的概率
        total_prob = sum([x[1] for x in results])
        if total_prob > 0:
            results = [[c, prob / total_prob] for c, prob in results]
        
        # 返回预测的类别和概率
        return results[0][0], results[0][1]

    def _get_class_prob(self, words, class_name):
        """计算 P(w1,w2..wn|x) * P(x)"""
        # P(x)
        p_x = self.p_class[class_name]
        # P(w1,w2..wn|x) = P(w1|x) * P(w2|x)...P(wn|x)
        p_w_x = self._get_words_class_prob(words, class_name)
        return p_x * p_w_x

    def _get_words_class_prob(self, words, class_name):
        """计算 P(w1|x) * P(w2|x)...P(wn|x)"""
        result = 1
        for word in words:
            unk_prob = self.word_class_prob[class_name]["<unk>"]
            result *= self.word_class_prob[class_name].get(word, unk_prob)
        return result

    def evaluate(self, test_data):
        """评估模型在测试集上的性能"""
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        for _, row in test_data.iterrows():
            true_label = str(row['label'])
            review = row['review']
            predicted_label, _ = self.predict(review)
            
            # 计算混淆矩阵
            if true_label == "1":
                if predicted_label == "1":
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if predicted_label == "1":
                    false_positive += 1
                else:
                    true_negative += 1
        
        # 计算性能指标
        accuracy = (true_positive + true_negative) / len(test_data) if len(test_data) > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n模型评估结果：")
        print(f"准确率(Accuracy): {accuracy:.4f}")
        print(f"精确率(Precision): {precision:.4f}")
        print(f"召回率(Recall): {recall:.4f}")
        print(f"F1值(F1 Score): {f1_score:.4f}")
        print("\n混淆矩阵：")
        print(f"         预测正例  预测负例")
        print(f"实际正例    {true_positive}      {false_negative}")
        print(f"实际负例    {false_positive}      {true_negative}")
        
        return accuracy, precision, recall, f1_score

# 主函数
def main():
    # 初始化分类器
    classifier = BayesTextClassifier()
    
    # 加载训练集
    print("加载训练集...")
    train_data = classifier.load_data_from_csv("train.csv")
    
    # 训练模型
    print("开始训练模型...")
    classifier.train(train_data)
    
    # 加载测试集
    print("\n加载测试集...")
    test_data = classifier.load_data_from_csv("test.csv")
    
    # 评估模型
    print("开始评估模型...")
    classifier.evaluate(test_data)
    
    # 加载验证集
    print("\n加载验证集...")
    valid_data = classifier.load_data_from_csv("valid.csv")
    
    # 在验证集上评估
    print("开始在验证集上评估模型...")
    classifier.evaluate(valid_data)
    
    # 测试单个预测
    print("\n测试单个预测：")
    sample_text = "这个餐厅的食物非常好吃，服务也很周到！"
    predicted_label, prob = classifier.predict(sample_text)
    print(f"文本：{sample_text}")
    print(f"预测类别：{'正样本' if predicted_label == '1' else '负样本'}")
    print(f"预测概率：{prob:.4f}")
    
    sample_text = "这个餐厅的食物很难吃，服务也很差！"
    predicted_label, prob = classifier.predict(sample_text)
    print(f"\n文本：{sample_text}")
    print(f"预测类别：{'正样本' if predicted_label == '1' else '负样本'}")
    print(f"预测概率：{prob:.4f}")

if __name__ == "__main__":
    main()