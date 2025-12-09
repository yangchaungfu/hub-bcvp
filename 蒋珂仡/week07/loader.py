# -*- coding: utf-8 -*-

import json    #用来读写字典文件（保存词表）
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader   #一个数据集，一个数据加载器
from collections import Counter

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data, config, build_vocab=False):
        self.config = config
        self.data = data
        self.config["class_num"] = 2  # 0:差评, 1:好评

        # 词表构建逻辑：如果需要构建词表（通常是训练集）
        if build_vocab:
            self.build_vocab()
        else:
            self.vocab = load_vocab(config["vocab_path"])

        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def build_vocab(self):
        # 统计所有文本构建词表
        all_text = "".join([str(text) for text in self.data['review']])
        counts = Counter(all_text)
        vocab = {char: i + 2 for i, (char, _) in enumerate(counts.most_common(4000))}
        vocab['[PAD]'] = 0
        vocab['[UNK]'] = 1
        self.vocab = vocab
        # 保存词表到文件
        with open(self.config["vocab_path"], 'w', encoding='utf-8') as f:
            for key, value in self.vocab.items():
                f.write(json.dumps([key, value]) + "\n")

    def load(self):
        self.dataset = []
        for index, row in self.data.iterrows():
            review = str(row['review'])
            label = int(row['label'])

            input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.dataset.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def load_vocab(vocab_path):
    token_dict = {}
    try:
        with open(vocab_path, encoding="utf8") as f:
            for line in f:
                pair = json.loads(line)
                token_dict[pair[0]] = pair[1]
    except FileNotFoundError:
        # 如果文件还没生成，返回空字典，等待build_vocab生成
        return {}
    return token_dict


# 修改 load_data 函数，接受 DataFrame
def load_data(data_frame, config, shuffle=True, build_vocab=False):
    dg = DataGenerator(data_frame, config, build_vocab=build_vocab)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
