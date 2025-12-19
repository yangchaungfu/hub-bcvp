# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载 - 改为BERT版本
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                
                # 使用BERT tokenizer处理文本
                text = "".join(sentence)
                self.sentences.append(text)
                
                # 编码句子
                encoded = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,  # 添加[CLS]和[SEP]
                    max_length=self.config["max_length"],
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].squeeze()
                attention_mask = encoded['attention_mask'].squeeze()
                
                # 处理标签长度与BERT tokenizer一致
                processed_labels = self.process_labels_for_bert(sentence, labels, text)
                
                self.data.append([
                    input_ids,
                    attention_mask,
                    torch.LongTensor(processed_labels)
                ])
        return

    # 在loader.py的process_labels_for_bert方法中也需要相应修改
    def process_labels_for_bert(self, chars, labels, text):
        """
        处理标签以适应BERT tokenizer的分词结果
        """
        # 获取原始字符的标签
        char_labels = list(zip(chars, labels))
        
        # 使用BERT tokenizer对文本进行编码
        tokens = self.tokenizer.tokenize(text)
        processed_labels = []
        char_index = 0
        
        # 添加[CLS]标记的标签 (-1表示忽略)
        processed_labels.append(-1)
        
        for token in tokens:
            if char_index >= len(char_labels):
                processed_labels.append(-1)
                continue
                
            # 如果token以##开头，说明是子词
            if token.startswith("##"):
                # 子词继承前面字符的标签
                if len(processed_labels) > 0:
                    processed_labels.append(processed_labels[-1])
                else:
                    processed_labels.append(-1)
            else:
                if char_index < len(char_labels):
                    processed_labels.append(char_labels[char_index][1])
                    char_index += 1
                else:
                    processed_labels.append(-1)
                    
        # 添加[SEP]标记的标签 (-1表示忽略)
        processed_labels.append(-1)
        
        # 填充或截断到最大长度
        if len(processed_labels) > self.config["max_length"]:
            processed_labels = processed_labels[:self.config["max_length"]]
        else:
            processed_labels.extend([-1] * (self.config["max_length"] - len(processed_labels)))
            
        return processed_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train", Config)
