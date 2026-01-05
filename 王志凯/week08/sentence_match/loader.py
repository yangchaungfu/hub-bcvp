# -*- coding:utf-8 -*-

import json
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from config import Config
tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])

class SentenceMatchLoader(Dataset):
    def __init__(self, config, isTrain):
        super(SentenceMatchLoader, self).__init__()
        self.config = config
        self.isTrain = isTrain
        self.epoch_size = config["epoch_size"]
        self.data = []
        self.schema = load_schema(config["schema_path"])
        self.load()


    def load(self):
        self.q_vec_dict = dict()
        self.q_sent_dict = dict()
        data_list = []
        with open(self.config["train_data_path"], "r", encoding="utf-8") as f:
            for line in f:
                dict_data = json.loads(line)
                assert isinstance(dict_data, dict)
                data_list.append(dict_data)
                # 预先将所有问题与其标准问的索引进行映射
                questions, target = dict_data["questions"], dict_data["target"]
                for i, question in enumerate(questions):
                    seq = padding(question, self.config)
                    if self.config["model_type"] != "bert":
                        seq = torch.LongTensor(seq)
                    # target会重复，在后面加上序号，后续处理的时候再单独取出target
                    label = f"{self.schema[target]}_{i}"
                    self.q_vec_dict[label] = seq
                    self.q_sent_dict[label] = question
        if self.isTrain:
            for _ in range(self.epoch_size):
                # 双塔模型
                if self.config["train_type"] == "Siam":
                    if random.random() < 0.5:
                        # 随机正样本
                        sent1, sent2 = pick_pos_sample(data_list)
                        label = torch.LongTensor([1])
                    else:
                        # 随机负样本
                        sent1, sent2 = pick_neg_sample(data_list)
                        label = torch.LongTensor([-1])
                    # 将文本转化为序列
                    seq1, seq2 = padding(sent1, self.config), padding(sent2, self.config)
                    if self.config["model_type"] != "bert":
                        seq1, seq2 = torch.LongTensor(seq1), torch.LongTensor(seq2)
                    self.data.append([seq1, seq2, label])
                else:
                    # 三元组模型
                    a, p, n = pick_apn_sample(data_list)
                    a, p, n = padding(a, self.config), padding(p, self.config), padding(n, self.config)
                    if self.config["model_type"] != "bert":
                        a, p, n = torch.LongTensor(a), torch.LongTensor(p), torch.LongTensor(n)
                    self.data.append([a, p, n])
        else:
            # 加载测试数据
            with open(self.config["valid_data_path"], "r", encoding="utf-8") as f:
                for line in f:
                    list_data = json.loads(line)
                    assert isinstance(list_data, list)
                    question, target = list_data
                    seq = padding(question, self.config)
                    label = torch.LongTensor([self.schema[target]])
                    if self.config["model_type"] != "bert":
                        seq = torch.LongTensor(seq)
                    self.data.append([seq, label])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def padding(sent, config):
    max_len = config["max_length"]
    if config["model_type"] == "bert":
        # bert模型需要在首位加cls和sep
        sequence = tokenizer(sent,
                             padding='max_length',  # 补齐到最大长度
                             truncation=True,       # 大于最大长度就进行截断
                             max_length=max_len,    # 最大长度数值
                             return_tensors="pt")   # 以张量形式输出，后续就不需要额外转类型
    else:
        # tokenizer()返回：input_ids, attention_mask, token_type_ids；
        # 而tokenizer.encode()返回单一的embedding序列
        sequence = tokenizer.encode(sent,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=max_len,
                                    add_special_tokens=False)  # 是否使用cls sep特殊token(只有bert需要)
    return sequence

# 将target作为锚点
def pick_apn_sample(data_list):
    # 随机取两行数据
    random_ids = random.sample(range(len(data_list)), 2)
    # 将第一行的target作为锚点
    a = data_list[random_ids[0]]["target"]
    # 在锚点所在的行随机取一个question作为正样本
    p_questions = data_list[random_ids[0]]["questions"]
    p_idx = random.choice(range(len(p_questions)))
    p = p_questions[p_idx]
    # 在第二行随机取一个question作为负样本
    n_questions = data_list[random_ids[1]]["questions"]
    n_idx = random.choice(range(len(n_questions)))
    n = n_questions[n_idx]
    return a, p, n

def pick_pos_sample(data_list):
    random_idx = random.choice(range(len(data_list)))
    questions = data_list[random_idx]["questions"]
    if len(questions) < 2:
        return pick_pos_sample(data_list)
    sent1, sent2 = random.sample(questions, 2)
    return sent1, sent2

def pick_neg_sample(data_list):
    random_ids = random.sample(range(len(data_list)), 2)
    questions1 = data_list[random_ids[0]]["questions"]
    questions2 = data_list[random_ids[1]]["questions"]
    sent1, sent2 = random.choice(questions1), random.choice(questions2)
    return sent1, sent2


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

def load_data(config, isTrain=True):
    sml = SentenceMatchLoader(config, isTrain)
    if isTrain:
        data = DataLoader(sml, batch_size=config['batch_size'], shuffle=True)
    else:
        data = DataLoader(sml, batch_size=1000)
    return data



if __name__ == '__main__':
    train_data = load_data(Config, isTrain=True)
    print(train_data.dataset.__getitem__(0))

    valid_data = load_data(Config, isTrain=False)
    print(valid_data.dataset.__getitem__(0))