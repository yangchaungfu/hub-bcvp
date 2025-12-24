# -*- coding:utf-8 -*-

import random

import torch
from torch.utils.data import Dataset, DataLoader

class GenerativeLoader(Dataset):
    def __init__(self, config, data_path):
        super(GenerativeLoader, self).__init__()
        self.max_length = config["max_length"]
        self.sample_num = config["sample_num"]
        self.data_path = data_path
        self.vocab_dict = load_vocab(config["vocab_path"])
        self.corpus = load_corpus(data_path)
        self.data = []
        self.load()

    def load(self):
        for _ in range(self.sample_num):
            start = random.randint(0, len(self.corpus) - self.max_length - 1)
            end = start + self.max_length
            window = self.corpus[start:end]
            target = self.corpus[start + 1: end + 1]
            # 转序列
            sequence = sentence2sequence(window, self.vocab_dict, self.max_length)
            target = sentence2sequence(target, self.vocab_dict, self.max_length)
            self.data.append((torch.LongTensor(sequence), torch.LongTensor(target)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 补齐或截断（仅对列表或字符串进行操作）
def padding(seq, max_length, pad_token=0):
    assert isinstance(seq, (list, str))
    seq = seq[:max_length]
    if len(seq) < max_length:
        seq += [pad_token] * (max_length - len(seq))
    return seq

# 将文本转化为序列（附带补齐和截断）
def sentence2sequence(sentence, vocab, max_length):
    sequence = []
    unk_idx = vocab["[UNK]"]
    pad_idx = vocab["[PAD]"]
    sentence = sentence[: max_length]
    for char in sentence:
        idx = vocab.get(char, unk_idx)
        sequence.append(idx)
    if len(sentence) < max_length:
        count = max_length - len(sentence)
        sequence += [pad_idx] * count
    return sequence


def load_data(config, data_path):
    dataset = GenerativeLoader(config, data_path)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

def load_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for index, char in enumerate(f):
            vocab_dict[char.strip()] = index
    return vocab_dict

def load_corpus(data_path):
    corpus = ""
    with open(data_path, "r", encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus



if __name__ == '__main__':
    from config import Config
    # train_data = load_data(Config, Config["train_data_path"])
    # for data in train_data:
    #     print(data)
    #     break

    print(load_mask(Config))

