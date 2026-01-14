# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, logger):
        self.logger = logger
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(r"/mnt/2T/wwk/pycharm_environment/study/google-bert/bert-base-chinese")
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    #文本到对应的index
    #头尾分别加入[cls]和[sep]
    def encode_sentence(self, text, max_length, with_cls_token=True, with_sep_token=True):
        input_id = []
        if with_cls_token:
            input_id.append(self.tokenizer.vocab["[CLS]"])
        for char in text:
            input_id.append(self.tokenizer.vocab.get(char, self.tokenizer.vocab["[UNK]"]))
        if with_sep_token:
            input_id.append(self.tokenizer.vocab["[SEP]"])
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length, pad=None):
        input_id = input_id[:length]
        if pad is None:
            input_id += [self.tokenizer.vocab["[PAD]"]] * (length - len(input_id))
        else:
            input_id += [pad] * (length - len(input_id))
        return input_id

    #输入输出转化成序列
    def prepare_data(self, title, content):
        input_seq = self.encode_sentence(title, 512, False, True) #输入序列
        decode_input = self.encode_sentence(content, 512, False, False) #输出序列
        decode_output = self.encode_sentence(content, 512, False, True) #不进入模型，用于计算loss
        input_s = input_seq + decode_input
        output_s = [-100] * (len(input_seq) - 1) + decode_output
        input_s = self.padding(input_s, 150)
        output_s = self.padding(output_s, 150, -100)
        self.data.append([torch.LongTensor(input_s),
                          torch.LongTensor(output_s)])

        return


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, logger, shuffle=True):
    dg = DataGenerator(data_path, logger)
    dl = DataLoader(dg, batch_size=32, shuffle=shuffle)
    return dl



if __name__ == "__main__":
    dl = load_data(r'sample_data.json', logger=1)
    for x, y in dl:
        print(x.shape, y.shape)
        print(x[1], y[1])
        print(type(x.shape[0]))

