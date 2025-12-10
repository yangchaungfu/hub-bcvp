import json 
import re 
import os 
import torch 
import random 
import jieba 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from collections import defaultdict 

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None 
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[label].append(input_id)
                    
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line 
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = self.schema[label]
                    label_index = torch.LongTensor([label_index])
                    self.data.append([input_id, label_index])


    def encode_sentence(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            return self.data[index]

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test"
            return len(self.data)    
    
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        a1, a2 = random.sample(standard_question_index, 2)
        a = self.encode_sentence(a1)
        a = torch.LongTensor(a)
        p = random.choice(self.knwb[a1])
        n = random.choice(self.knwb[a2])
        return [a, p, n]
    
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path,'r',encoding="utf8") as f:
        for index,line in enumerate(f):
            word = line.strip()
            vocab[word] = index + 1
    return vocab

def load_schema(schema_path):
    with open(schema_path, 'r', encoding='utf8') as f:
        return json.loads(f.read())
    
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"])
    return dl 