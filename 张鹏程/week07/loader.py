import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        
        # 尝试从JSON文件加载标签映射（由split_csv_data生成）
        label_map_path = os.path.join(os.path.dirname(data_path), "label_mapping.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf8') as f:
                label_info = json.load(f)
                self.index_to_label = {int(k): str(v) for k, v in label_info["index_to_label"].items()}
                self.label_to_index = {str(k): int(v) for k, v in label_info["label_to_index"].items()}
        else:
            # 兼容旧的标签映射（用于已有的JSON数据）
            self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                                   5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                                   10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                                   14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
            self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        
        self.config["class_num"] = len(self.index_to_label)
        
        if self.config["model_type"] in ["bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = str(line["tag"])  # 确保tag是字符串
                label = self.label_to_index.get(tag)
                if label is None:
                    # 如果标签不在映射中，跳过该数据或使用默认标签
                    continue
                title = line["title"]
                label_index = torch.LongTensor([label])
                
                if self.config["model_type"] in ["bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]:
                    # 使用tokenizer返回input_ids和attention_mask
                    encoded = self.tokenizer.encode_plus(
                        title, 
                        max_length=self.config["max_length"],
                        padding='max_length',
                        truncation=True,
                        return_tensors=None  # 返回列表而不是tensor
                    )
                    input_id = torch.LongTensor(encoded['input_ids'])
                    attention_mask = torch.LongTensor(encoded['attention_mask'])
                    self.data.append([input_id, attention_mask, label_index])
                else:
                    input_id = self.encode_sentence(title)
                    input_id = torch.LongTensor(input_id)
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def split_csv_data(csv_path, output_dir="./data", random_seed=987):
    """
    将CSV文件按7:2:1的比例随机分成训练集、验证集、测试集
    并将数据转换为JSON格式，保存标签映射
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        random_seed: 随机种子
    
    Returns:
        train_path, valid_path, test_path: 三个数据集的路径
    """
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("CSV数据分割功能需要安装pandas和scikit-learn: pip install pandas scikit-learn\n"
                         "如果数据已经分割好，可以跳过此步骤。")
    
    # 读取CSV文件
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf8')
    
    # 获取标签列和文本列（假设第一列是标签，第二列是文本）
    label_col = df.columns[0]
    text_col = df.columns[1]
    
    # 获取所有唯一的标签并创建映射
    unique_labels = sorted(df[label_col].unique())
    label_to_index = {str(label): idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: str(label) for label, idx in label_to_index.items()}
    
    # 保存标签映射
    os.makedirs(output_dir, exist_ok=True)
    label_map = {
        "label_to_index": label_to_index,
        "index_to_label": index_to_label
    }
    label_map_path = os.path.join(output_dir, "label_mapping.json")
    with open(label_map_path, 'w', encoding='utf8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    # 准备数据：转换为字符串格式的标签
    df['tag'] = df[label_col].astype(str)
    df['title'] = df[text_col].astype(str)
    
    # 第一次分割：70%训练，30%临时（验证+测试）
    try:
        train_df, temp_df = train_test_split(
            df[['tag', 'title']], 
            test_size=0.3, 
            random_state=random_seed,
            stratify=df['tag']  # 保持标签分布
        )
    except ValueError:
        # 如果某个类别样本太少无法分层，则使用普通分割
        train_df, temp_df = train_test_split(
            df[['tag', 'title']], 
            test_size=0.3, 
            random_state=random_seed
        )
    
    # 第二次分割：从30%中分出20%验证，10%测试（即整体的2:1）
    try:
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=1/3,  # 1/3的temp是测试集，2/3是验证集
            random_state=random_seed,
            stratify=temp_df['tag']
        )
    except ValueError:
        # 如果某个类别样本太少无法分层，则使用普通分割
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=1/3,
            random_state=random_seed
        )
    
    # 转换为JSON格式（每行一个JSON对象）
    def df_to_json(df, output_path):
        with open(output_path, 'w', encoding='utf8') as f:
            for _, row in df.iterrows():
                json_obj = {"tag": row['tag'], "title": row['title']}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    # 保存三个数据集
    train_path = os.path.join(output_dir, "train_tag_news.json")
    valid_path = os.path.join(output_dir, "valid_tag_news.json")
    test_path = os.path.join(output_dir, "test_tag_news.json")
    
    df_to_json(train_df, train_path)
    df_to_json(valid_df, valid_path)
    df_to_json(test_df, test_path)
    
    print(f"\n数据分割完成！")
    print(f"训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(valid_df)} 条 ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    print(f"总数据: {len(df)} 条")
    print(f"标签类别数: {len(unique_labels)}")
    print(f"标签映射已保存到: {label_map_path}\n")
    
    return train_path, valid_path, test_path


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    # 测试CSV数据分割
    csv_path = "文本分类练习.csv"
    if os.path.exists(csv_path):
        split_csv_data(csv_path, output_dir="./data")
    
    # 测试数据加载
    if os.path.exists("./data/train_tag_news.json"):
        dg = DataGenerator("./data/train_tag_news.json", Config)
        print(f"加载了 {len(dg)} 条训练数据")
        print(f"样本示例: {dg[0]}")
