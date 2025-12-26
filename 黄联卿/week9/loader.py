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
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用 BERT 的 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"], do_basic_tokenize=False)
        self.config["vocab_size"] = self.tokenizer.vocab_size
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.char_lengths = []  # 保存每个样本的原始字符数
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                if not segment.strip():
                    continue
                chars = []
                labels = []
                lines = segment.strip().split("\n")
                for line in lines:
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        char, label = parts[0], parts[-1]
                        chars.append(char)
                        labels.append(self.schema[label])
                if not chars:
                    continue

                # === 关键：对齐 tokenizer ===
                token_ids = []
                label_ids = []

                # 添加 [CLS]
                token_ids.append(self.tokenizer.cls_token_id)
                label_ids.append(-1)  # ignore

                # 逐字处理
                for char, label in zip(chars, labels):
                    # tokenize 单个字符
                    tokens = self.tokenizer.tokenize(char)
                    token_ids.extend(self.tokenizer.convert_tokens_to_ids(tokens))
                    # 如果这个字被拆成多个 subword，第一个用真实标签，其余用 -1 或 I-label（但你数据是字符级，通常不会拆）
                    label_ids.append(label)
                    # 如果被拆（理论上中文极少），后面 subword 设为 -1
                    label_ids.extend([-1] * (len(tokens) - 1))

                # 添加 [SEP]
                token_ids.append(self.tokenizer.sep_token_id)
                label_ids.append(-1)

                # 截断或填充到 max_length
                if len(token_ids) > self.config["max_length"]:
                    token_ids = token_ids[:self.config["max_length"]]
                    label_ids = label_ids[:self.config["max_length"]]
                else:
                    pad_len = self.config["max_length"] - len(token_ids)
                    token_ids += [self.tokenizer.pad_token_id] * pad_len
                    label_ids += [-1] * pad_len  # pad 位置也忽略

                self.sentences.append("".join(chars))  # 保存原始句子用于 eval
                self.data.append([
                    torch.LongTensor(token_ids),
                    torch.LongTensor(label_ids)
                ])
                self.char_lengths.append(len(chars))
        return

    # def encode_sentence(self, text, padding=True):
    #     # 使用 BERT tokenizer 编码
    #     if isinstance(text, list):
    #         text = "".join(text)
    #
    #     encoding = self.tokenizer(
    #         text,
    #         max_length=self.config["max_length"],
    #         truncation=True,
    #         padding="max_length" if padding else False,
    #         return_tensors="pt"  # 返回 PyTorch tensor
    #     )
    #     return encoding["input_ids"].squeeze().tolist()

    #补齐或截断输入的序列，使其可以在一个batch内运算
    # def padding(self, input_id, pad_token=0):
    #     input_id = input_id[:self.config["max_length"]]
    #     input_id += [pad_token] * (self.config["max_length"] - len(input_id))
    #     return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
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

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

