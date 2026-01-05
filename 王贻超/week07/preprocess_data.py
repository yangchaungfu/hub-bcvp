# -*- coding: utf-8 -*-
import pandas as pd
import json
import random
import os

def convert_csv_to_json(csv_path, train_ratio=0.8):
    """
    将CSV格式的电商评论数据转换为JSON格式，并划分为训练集和验证集
    """
    # 确保数据目录存在
    data_dir = "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 数据清洗
    df = df.dropna()
    df.columns = ['label', 'title']  # 重命名列
    
    # 按标签分组
    positive_data = df[df['label'] == 1].to_dict('records')
    negative_data = df[df['label'] == 0].to_dict('records')
    
    # 打乱数据
    random.shuffle(positive_data)
    random.shuffle(negative_data)
    
    # 划分训练集和验证集
    train_data = []
    valid_data = []
    
    # 正样本划分
    pos_train_count = int(len(positive_data) * train_ratio)
    for item in positive_data[:pos_train_count]:
        train_data.append({"tag": "positive", "title": item['title']})
    for item in positive_data[pos_train_count:]:
        valid_data.append({"tag": "positive", "title": item['title']})
    
    # 负样本划分
    neg_train_count = int(len(negative_data) * train_ratio)
    for item in negative_data[:neg_train_count]:
        train_data.append({"tag": "negative", "title": item['title']})
    for item in negative_data[neg_train_count:]:
        valid_data.append({"tag": "negative", "title": item['title']})
    
    # 保存训练集
    with open("../data/train_review.json", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证集
    with open("../data/valid_review.json", 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据预处理完成:")
    print(f"训练集样本数: {len(train_data)} (正样本: {len([x for x in train_data if x['tag']=='positive'])}, 负样本: {len([x for x in train_data if x['tag']=='negative'])})")
    print(f"验证集样本数: {len(valid_data)} (正样本: {len([x for x in valid_data if x['tag']=='positive'])}, 负样本: {len([x for x in valid_data if x['tag']=='negative'])})")

def analyze_data(csv_path):
    """
    分析数据集统计信息
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.columns = ['label', 'title']
    
    total_count = len(df)
    positive_count = sum(df['label'] == 1)
    negative_count = sum(df['label'] == 0)
    
    # 计算文本长度统计
    df['text_length'] = df['title'].apply(len)
    avg_length = df['text_length'].mean()
    max_length = df['text_length'].max()
    min_length = df['text_length'].min()
    
    print("=" * 50)
    print("电商评论数据集统计信息:")
    print("=" * 50)
    print(f"总样本数: {total_count}")
    print(f"正样本数(好评): {positive_count} ({positive_count/total_count*100:.2f}%)")
    print(f"负样本数(差评): {negative_count} ({negative_count/total_count*100:.2f}%)")
    print(f"文本平均长度: {avg_length:.2f}")
    print(f"文本最大长度: {max_length}")
    print(f"文本最小长度: {min_length}")
    print("=" * 50)
    
    return {
        "total": total_count,
        "positive": positive_count,
        "negative": negative_count,
        "avg_length": avg_length,
        "max_length": max_length,
        "min_length": min_length
    }

if __name__ == "__main__":
    # 数据分析
    stats = analyze_data('文本分类练习.csv')
    
    # 数据预处理
    convert_csv_to_json('文本分类练习.csv')
