# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()
        # 仅当加载训练集时，才处理valid_keys并校验
        if self.data_type == "train":
            self.valid_keys = [k for k in self.knwb.keys() if len(self.knwb[k]) >= 1]
            # 兜底校验：确保valid_keys非空
            if not self.valid_keys:
                raise ValueError("self.knwb中没有有效的标准问题组，请检查数据！")
        else:
            self.valid_keys = []

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                #加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]

    #依照一定概率生成负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):

        if self.data_type != "train":
            raise RuntimeError("random_train_sample仅支持训练集调用！")

        # 步骤1：循环选择正样本组（确保组内至少有2个样本，避免递归）
        positive_key = None
        while positive_key is None:
            candidate_key = random.choice(self.valid_keys)
            if len(self.knwb[candidate_key]) >= 2:
                positive_key = candidate_key

        # 步骤2：从正样本组中选2个样本作为s1（锚点）和s2（正样本）
        s1, s2 = random.sample(self.knwb[positive_key], 2)  # 返回两个单个张量

        # 步骤3：选择负样本组（与正样本组不同）
        negative_keys = [k for k in self.valid_keys if k != positive_key]
        # 极端情况：只有一个组（理论上不应出现，仅兜底）
        if not negative_keys:
            s3 = random.choice(self.knwb[positive_key])
        else:
            negative_key = random.choice(negative_keys)
            # 用random.choice替代random.sample，返回单个张量（而非列表）
            s3 = random.choice(self.knwb[negative_key])

        # 返回三个单个张量，无列表嵌套
        return [s1, s2, s3]
        """
        standard_question_index = list(self.knwb.keys())
        #随机正样本
        # if random.random() <= self.config["positive_sample_rate"]:
        while True:
            # 重新选择p，确保p+1对应的列表非空
            p = random.choice(list(self.knwb.keys()))
            # 检查self.knwb中是否有p+1这个键，且对应的列表非空
            if (p + 1) in self.knwb and len(self.knwb[p + 1]) >= 1:
                break
        # p = random.choice(standard_question_index)
        #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
            s3 = random.sample(self.knwb[p-1], 1)
            return [s1, s2, s3]
        #随机负样本
    
        else:
            p, n = random.sample(standard_question_index, 2)
            s1 = random.choice(self.knwb[p])
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([-1])]
    """


#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
        # 确保[UNK]存在（未知字符/词的索引）
    if "[UNK]" not in token_dict:
        token_dict["[UNK]"] = len(token_dict) + 1
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

# 自定义collate_fn：将批量的张量列表拼接成批量张量
def collate_fn(batch):
    if len(batch[0]) == 3:
        # 训练集：[s1, s2, s3]
        s1_batch = torch.stack([item[0] for item in batch])
        s2_batch = torch.stack([item[1] for item in batch])
        s3_batch = torch.stack([item[2] for item in batch])
        return s1_batch, s2_batch, s3_batch
    else:
        # 测试集：[input_id, label_index]
        input_batch = torch.stack([item[0] for item in batch])
        label_batch = torch.stack([item[1] for item in batch])
        return input_batch, label_batch


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    # dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    dl = DataLoader(
        dg,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn  # 关键：添加自定义拼接函数
    )
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/valid.json", Config)
    print(dg[1])
