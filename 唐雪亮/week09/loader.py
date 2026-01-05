# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

"""
数据加载
"""


class BertNERDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        self.schema = self._load_schema()
        self.id2label = {v: k for k, v in self.schema.items()}  # id→标签映射
        self.sentences = []
        self.data = self._load_and_process_data(data_path)


    def _load_schema(self):
        """加载标签映射表（label→id）"""
        with open(self.config["schema_path"], encoding="utf8") as f:
            return json.load(f)

    def _load_and_process_data(self, data_path):
        """加载数据并处理BERT分词+标签对齐"""
        data = []
        with open(data_path, encoding="utf8") as f:
            # 按空行分割句子
            sentence_segments = f.read().split("\n\n")
            for segment in sentence_segments:
                if not segment.strip():
                    continue
                # 解析字符和标签
                chars, label_ids = [], []
                for line in segment.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    char, label = line.split(maxsplit=1)
                    chars.append(char)
                    label_ids.append(self.schema[label])
                self.sentences.append(" ".join(chars))
                # 处理BERT输入和标签对齐
                input_ids, attention_mask, aligned_labels = self._align_bert_input(chars, label_ids)
                data.append({
                    "input_ids": torch.LongTensor(input_ids),
                    "attention_mask": torch.LongTensor(attention_mask),
                    "labels": torch.LongTensor(aligned_labels)
                })
        return data

    def _align_bert_input(self, chars, label_ids):
        """
        核心：处理BERT分词与原始字符的标签对齐
        完全基于offset_mapping推导，不依赖word_ids
        """
        # 1. 拼接原始字符为完整文本（用于分词对齐）
        raw_text = "".join(chars)

        # 2. BERT分词（必须返回offset_mapping）
        tokenized = self.tokenizer(
            raw_text,
            add_special_tokens=True,  # 添加[CLS] [SEP]
            max_length=self.config["max_length"],
            padding="max_length",  # 填充到最大长度
            truncation=True,  # 截断超长文本
            return_offsets_mapping=True,  # 关键：返回每个token的字符偏移
            return_tensors="pt"  # 返回tensor（后续需展平）
        )

        # 展平tensor（去除batch维度）
        input_ids = tokenized["input_ids"].squeeze(0).tolist()
        attention_mask = tokenized["attention_mask"].squeeze(0).tolist()
        offset_mapping = tokenized["offset_mapping"].squeeze(0).tolist()

        # 3. 初始化对齐后的标签（-100表示忽略该位置的loss）
        aligned_labels = [-100] * len(input_ids)

        # 4. 遍历offset_mapping，映射原始字符标签到分词后token
        char_idx = 0  # 原始字符的索引
        token_idx = 0  # 分词后token的索引

        for token_idx, (start, end) in enumerate(offset_mapping):
            # 跳过特殊标记（[CLS]/[SEP]/padding，offset为(0,0)）
            if start == 0 and end == 0:
                continue

            # 找到当前token对应的原始字符索引
            # 处理subword：只给每个原始字符的第一个subword分配标签
            if start == char_idx:
                if char_idx < len(label_ids):
                    aligned_labels[token_idx] = label_ids[char_idx]
                char_idx = end  # 移动到下一个原始字符

        return input_ids, attention_mask, aligned_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_data(data_path, config, shuffle=True):
    """创建数据加载器"""
    dataset = BertNERDataset(data_path, config)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle
    )



if __name__ == "__main__":
    from config import Config
    dg = BertNERDataset("ner_data/test", Config)
    print(len(dg))
    print(dg[0])
    d1 = load_data("ner_data/test", Config)
    print(len(d1))
    valid_data = load_data(Config["valid_data_path"], Config, shuffle=False)
    print(valid_data.dataset.__getitem__(0))

