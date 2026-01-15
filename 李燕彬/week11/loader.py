#coding:utf8

import torch
import random
import json



#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#max_input_len 最大输入长度
#max_output_len 最大输出长度
#data_list 数据列表，每个元素包含title和content

def build_dataset(sample_length, vocab, max_input_len, max_output_len, data_list):
    dataset_x = []
    dataset_y = []
    
    # 打乱数据顺序
    random.shuffle(data_list)
    
    for i in range(sample_length):
        # 循环取数据，确保有足够的样本
        data = data_list[i % len(data_list)]
        x, y = build_sample(vocab, max_input_len, max_output_len, data["title"], data["content"])
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


#生成一个样本
#将title作为输入，content作为输出

def build_sample(vocab, max_input_len, max_output_len, title, content):
    # 处理输入序列（title）
    x = [vocab.get(char, vocab["<UNK>"]) for char in title[:max_input_len]]
    # 处理输出序列（content）
    y = [vocab.get(char, vocab["<UNK>"]) for char in content[:max_output_len]]
    
    # 填充或截断到固定长度
    x = pad_sequence(x, max_input_len, vocab["<pad>"])
    y = pad_sequence(y, max_output_len, vocab["<pad>"])
    
    return x, y


#序列填充或截断函数
def pad_sequence(seq, max_len, pad_id):
    if len(seq) < max_len:
        # 填充到指定长度
        seq += [pad_id] * (max_len - len(seq))
    else:
        # 截断到指定长度
        seq = seq[:max_len]
    return seq


#加载字表
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index
    # 映射标准特殊标记
    vocab["<pad>"] = vocab.get("[PAD]", 0)
    vocab["<UNK>"] = vocab.get("[UNK]", 1)
    vocab["<s>"] = vocab.get("[CLS]", 2)
    vocab["</s>"] = vocab.get("[SEP]", 3)
    return vocab


#加载JSON语料
def load_corpus(path):
    data_list = []
    with open(path, encoding="utf8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    if "title" in data and "content" in data:
                        data_list.append(data)
                except json.JSONDecodeError:
                    continue
    return data_list