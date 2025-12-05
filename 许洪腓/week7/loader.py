import json 
import re 
import os 
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from transformers import BertTokenizer 

class DataGenerator:
    def __init__(self, data_path, config, train_ratio, is_train):
        self.config = config 
        self.path = data_path
        self.train_ratio = train_ratio
        self.is_train = is_train
        self.index_to_label = {0:'差评',1:'好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()
    
    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for idx,line in enumerate(f):
                if idx<1:
                    continue
                line = line.split(',')
                label, review = int(line[0]), line[1]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        split_idx = int(len(self.data)*self.train_ratio)
        if self.is_train:
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
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
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1 
    return token_dict 

def load_data(data_path, config, shuffle=True, is_train=True):
    dg = DataGenerator(data_path, config, config["train_ratio"], is_train)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl 

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(Config["train_data_path"], Config)
    print(dg[1])