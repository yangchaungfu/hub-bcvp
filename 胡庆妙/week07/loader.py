# coding:utf8

import json
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator(Dataset):
    def __init__(self, path, config):
        super().__init__()  # 调用父类初始化
        self.path = path
        self.config = config

        self.index_to_label = {0: '0', 1: '1'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())

        self.config["class_num"] = len(self.index_to_label)  # 类别数

        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self._load_data()

    def _load_data(self):
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            f.readline()  # 读取并忽略第一行，因为第一行是数据的描述性文字
            for line in f:
                line = line.strip()  # strip() 移除行尾换行符
                label = line[0:1]
                label_id = self.label_to_index[label]
                review = line[2:]
                if self.config["model_type"] == "bert":
                    input_ids = self.tokenizer.encode(review, max_length=self.config["sentence_len"],
                                                      padding="max_length", truncation=True)
                else:
                    input_ids = self.encode_text(review, self.config["sentence_len"])
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor([label_id])])
        return

    def encode_text(self, text, max_length):
        input_ids = []
        for char in text:
            input_ids.append(self.vocab.get(char, self.vocab["<unk>"]))

        input_ids = input_ids[:max_length]  # 截断
        input_ids += [0] * (max_length - len(input_ids))  # 补齐
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    char_dict = {'<pad>': 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            char_dict[char] = index + 1  # 0留给padding位置，所以从1开始
    char_dict['<unk>'] = len(char_dict)
    return char_dict


# 用torch自带的DataLoader类封装数据
def load_data(path, config, shuffle=True):
    data_gen = DataGenerator(path, config)
    data_loader = DataLoader(data_gen, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader


if __name__ == "__main__":
    from config import Config

    data_gen = DataGenerator(Config["train_data_path"], Config)
    print(data_gen[1])
    print(data_gen[2])
    print(data_gen[3])
