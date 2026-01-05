# -*- coding: utf-8 -*-

import json

import jieba
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用BERT tokenizer
        self.use_bert = "pretrain_model_path" in config and config["pretrain_model_path"]
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
            self.config["vocab_size"] = len(self.tokenizer.vocab)
        else:
            # 使用自定义vocab（已注释，改用BERT）
            # self.vocab = load_vocab(config["vocab_path"])
            # self.config["vocab_size"] = len(self.vocab)
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                if self.use_bert:
                    # 使用BERT tokenizer编码
                    input_ids, aligned_labels = self.encode_sentence_with_bert(sentenece, labels)
                    input_ids = self.padding(input_ids, self.tokenizer.pad_token_id)
                    aligned_labels = self.padding(aligned_labels, -1)
                else:
                    # 使用自定义vocab编码（已注释，改用BERT）
                    input_ids = self.encode_sentence(sentenece)
                    labels = self.padding(labels, -1)
                    aligned_labels = labels
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(aligned_labels)])
        return

    def encode_sentence_with_bert(self, sentence_chars, labels):
        """
        使用BERT tokenizer编码句子，并处理标签对齐
        BERT会添加[CLS]和[SEP]，可能对字符进行subword切分
        """
        # 将字符列表转换为字符串
        sentence = "".join(sentence_chars)

        # 使用tokenizer编码，获取token ids和offset mapping
        encoded = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            max_length=self.config["max_length"],
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors=None
        )

        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]

        # 对齐标签：将字符级别的标签对齐到token级别
        # offset_mapping中每个元素是(start, end)，表示token在原始字符串中的字符位置
        aligned_labels = []
        # 记录每个字符是否已经被分配标签（处理subword切分的情况）
        char_label_assigned = [False] * len(sentence)

        for i, (start, end) in enumerate(offset_mapping):
            # [CLS]和[SEP]标记，offset为(0, 0)或超出句子范围，标签设为-1（忽略）
            # 对于[SEP]，可能是(len(sentence), len(sentence))或(0, 0)
            if start == 0 and end == 0:
                # [CLS]或[SEP]标记
                aligned_labels.append(-1)
            elif start >= len(sentence) or end > len(sentence):
                # [SEP]标记或超出范围（可能是截断导致的）
                aligned_labels.append(-1)
            elif start < len(sentence) and end > start:
                # 有效的内容token
                # 找到这个token对应的第一个字符位置
                # 对于中文BERT，通常一个字符对应一个token，但可能有subword切分
                # 如果多个token对应同一个字符，只给第一个token分配标签
                char_pos = start
                if char_pos < len(labels):
                    # 如果这个字符还没有被分配标签，则分配
                    if not char_label_assigned[char_pos]:
                        aligned_labels.append(labels[char_pos])
                        char_label_assigned[char_pos] = True
                    else:
                        # 这个字符已经被前面的token分配了标签（subword情况），设为-1忽略
                        aligned_labels.append(-1)
                else:
                    # 超出标签范围（可能是截断导致的）
                    aligned_labels.append(-1)
            else:
                aligned_labels.append(-1)

        return input_ids, aligned_labels

    def encode_sentence(self, text, padding=True):
        """
        使用自定义vocab编码（已注释，改用BERT）
        """
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        """
        补齐或截断序列到max_length
        pad_token: 对于input_ids使用tokenizer的pad_token_id，对于labels使用-1
        """
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)
