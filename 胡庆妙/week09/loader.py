# -*- coding: utf-8 -*-

import json
import torch
import jieba
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator(Dataset):
    def __init__(self, data_path, config, vocab):
        self.path = data_path
        self.config = config
        self.vocab = vocab

        self.sentences = []  # 从语料转换得来的自然语句
        self.schema = self.load_schema(config["schema_path"])
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:  # 每段对应一个句子
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)  # 语句中的字
                    labels.append(self.schema[label])  # 语句中字的标注 -> 类别, 9类标注映射为9个类别
                self.sentences.append("".join(sentenece))
                input_ids = self.encode_sentence(sentenece)
                labels = self.padding(labels, -1)

                # [一句自然语句的字序列, 一句自然语句的标注序列]
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_ids = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_ids.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_ids = self.padding(input_ids)
        return input_ids

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["sentence_len"]]
        input_id += [pad_token] * (self.config["sentence_len"] - len(input_id))
        return input_id

    @staticmethod
    def load_schema(path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    vocab = {"[PAD]": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            vocab[token] = index + 1  # 0留给padding位置，所以从1开始
    return vocab


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, vocab, shuffle=True):
    dg = DataGenerator(data_path, config, vocab)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    vocab = load_vocab(Config["vocab_path"])

    load_data(Config["train_data_path"], Config, vocab)
