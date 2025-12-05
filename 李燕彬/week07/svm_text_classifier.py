import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from text_preprocessor import TextPreprocessor

class SVMTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = SVC(kernel='linear', C=1.0)
        self.preprocessor = TextPreprocessor()

    def train(self, train_csv_path):
        """训练SVM分类器"""
        print("加载训练数据...")
        train_reviews, train_labels = self.preprocessor.load_data(train_csv_path)
        
        print("分词处理...")
        # 使用jieba分词并转换为空格分隔的字符串
        train_corpus = []
        for review in train_reviews:
            tokens = jieba.lcut(review)
            train_corpus.append(' '.join(tokens))
        
        print("向量化文本...")
        # 训练TF-IDF向量器并转换训练集
        X_train = self.vectorizer.fit_transform(train_corpus)
        
        print("训练SVM分类器...")
        # 训练SVM分类器
        self.classifier.fit(X_train, train_labels)
        
        print("训练完成！")
        return

    def evaluate(self, test_csv_path):
        """在测试集上评估模型"""
        print(f"\n加载{test_csv_path}数据...")
        test_reviews, test_labels = self.preprocessor.load_data(test_csv_path)
        
        print("分词处理...")
        # 使用jieba分词并转换为空格分隔的字符串
        test_corpus = []
        for review in test_reviews:
            tokens = jieba.lcut(review)
            test_corpus.append(' '.join(tokens))
        
        print("向量化文本...")
        # 使用训练好的向量器转换测试集
        X_test = self.vectorizer.transform(test_corpus)
        
        print("预测结果...")
        # 预测
        y_pred = self.classifier.predict(X_test)
        
        # 评估
        accuracy = accuracy_score(test_labels, y_pred)
        report = classification_report(test_labels, y_pred)
        
        print(f"准确率: {accuracy:.4f}")
        print("分类报告:")
        print(report)
        
        return accuracy

    def predict(self, text):
        """预测单个文本的类别"""
        # 分词
        tokens = jieba.lcut(text)
        corpus = [' '.join(tokens)]
        
        # 向量化
        X = self.vectorizer.transform(corpus)
        
        # 预测
        predicted_label = self.classifier.predict(X)[0]
        
        return predicted_label

def main():
    # 创建分类器实例
    svm_classifier = SVMTextClassifier()
    
    # 训练模型
    print("=== SVM文本分类器训练 ===")
    svm_classifier.train("train.csv")
    
    # 评估模型
    print("\n=== 在测试集上评估 ===")
    svm_classifier.evaluate("test.csv")
    
    print("\n=== 在验证集上评估 ===")
    svm_classifier.evaluate("valid.csv")
    
    # 测试单个预测
    print("\n=== 测试单个预测 ===")
    sample_texts = [
        "这个餐厅的食物非常好吃，服务也很周到！",
        "这个餐厅的食物很难吃，服务也很差！"
    ]
    
    for text in sample_texts:
        predicted_label = svm_classifier.predict(text)
        print(f"文本：{text}")
        print(f"预测类别：{'正样本' if predicted_label == 1 else '负样本'}")

if __name__ == "__main__":
    main()