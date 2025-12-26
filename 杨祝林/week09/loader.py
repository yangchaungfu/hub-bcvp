# -*- coding: utf-8 -*-
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast  # 引入BERT的Fast tokenizer

"""
数据加载（适配BERT+CRF的NER任务）
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        # 加载BERT的tokenizer（核心替换：去掉自定义vocab，改用BERT tokenizer）
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        # 加载标签映射（schema：label -> id，如"B-ORG": 1）
        self.schema = self.load_schema(config["schema_path"])
        # 反向映射：id -> label（后续实体提取用）
        self.id2label = {v: k for k, v in self.schema.items()}
        # 记录O标签的id（用于填充[CLS]位置的标签）
        self.o_label_id = self.schema.get("O", 0)
        # 存储原始句子（字列表）和标签列表
        self.sentences = []  # 元素：字列表，如["他", "说", "中"]
        self.labels_list = []  # 元素：标签字符串列表，如["O", "O", "B-ORG"]
        # 加载数据
        self.load()

    def load(self):
        """
        读取数据：每行字+标签，空行分隔句子
        """
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence_chars = []  # 存储单个句子的字
                sentence_labels = []  # 存储单个句子的标签（字符串）
                for line in segment.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # 分割字和标签（处理多个空格的情况）
                    parts = line.split()
                    if len(parts) >= 2:
                        char, label = parts[0], parts[1]
                        sentence_chars.append(char)
                        sentence_labels.append(label)
                if not sentence_chars:  # 跳过空句子
                    continue
                # 存储原始数据
                self.sentences.append(sentence_chars)
                self.labels_list.append(sentence_labels)
                # 编码句子和标签（适配BERT）
                encoding, label_ids = self.encode_sentence(sentence_chars, sentence_labels)
                # 转换为tensor并存储
                input_ids = torch.LongTensor(encoding["input_ids"])
                attention_mask = torch.LongTensor(encoding["attention_mask"])
                label_ids = torch.LongTensor(label_ids)
                # self.data.append([input_ids, attention_mask, label_ids])
                self.data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids
                })
        return

    def encode_sentence(self, sentence_chars, sentence_labels):
        """
        用BERT tokenizer编码句子，并对齐标签
        参数：
            sentence_chars: 字列表，如["他", "说", "中"]
            sentence_labels: 标签字符串列表，如["O", "O", "B-ORG"]
        返回：
            encoding: tokenizer的编码结果（input_ids, attention_mask）
            label_ids: 对齐后的标签id列表（长度=max_length）
        """
        # 1. BERT编码：is_split_into_words=True表示输入是已分词的列表（这里是单字）
        encoding = self.tokenizer(
            sentence_chars,
            is_split_into_words=True,
            max_length=self.config["max_length"],
            padding="max_length",  # 补齐到max_length
            truncation=True,       # 超过max_length则截断
            return_tensors="pt"    # 返回tensor
        )
        # 移除batch维度（tokenizer返回的是(1, max_length)，需要变为(max_length,)）
        encoding = {k: v.squeeze(0).tolist() for k, v in encoding.items()}
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 2. 标签对齐：处理[CLS]、[SEP]、padding
        label_ids = [self.o_label_id] + [-1] * (self.config["max_length"] - 1)  # [CLS]设为O标签，其余初始化为-1
        # 获取word_ids：映射token到原句子的字索引（None表示[CLS]、[SEP]、padding）
        word_ids = self.tokenizer(
            sentence_chars,
            is_split_into_words=True,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True
        ).word_ids()

        # 填充标签（跳过[CLS]、[SEP]、padding）
        for i in range(len(word_ids)):
            word_id = word_ids[i]
            # 跳过[CLS]（i=0）、[SEP]、padding，或超出原标签长度的部分
            if i == 0 or word_id is None or word_id >= len(sentence_labels):
                continue
            # 填充对应的标签id
            label_ids[i] = self.schema[sentence_labels[word_id]]

        # 重新构造encoding（只保留input_ids和attention_mask）
        encoding = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        return encoding, label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        返回单条数据：(input_ids, attention_mask, label_ids)
        """
        return self.data[index]

    def load_schema(self, path):
        """
        加载标签映射表（json文件，格式：{"O":0, "B-ORG":1, ...}）
        """
        with open(path, encoding="utf8") as f:
            return json.load(f)


#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict



# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    # 测试用的配置（替换为你的实际配置）
    config = {
        "bert_model_name": "bert-base-chinese",  # 中文BERT模型
        "max_length": 128,  # 句子最大长度
        "batch_size": 2,     # 批次大小
        "schema_path": "schema.json"  # 标签映射文件路径
    }
    # 测试数据加载
    dg = DataGenerator("ner_data/train", config)
    print(f"数据条数：{len(dg)}")
    # 查看第一条数据
    input_ids, attention_mask, label_ids = dg[0]
    print(f"input_ids形状：{input_ids.shape}")
    print(f"attention_mask形状：{attention_mask.shape}")
    print(f"label_ids形状：{label_ids.shape}")