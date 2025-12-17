# -*- coding:utf-8 -*-

import os
import torch
from config import Config, Labels
from model import SequenceLabelModel
from loader import sentence2sequence, load_vocab
from transformers import BertConfig


def load_model(model_base_path):
    bertConfig = BertConfig.from_pretrained(Config["pretrain_model_path"])
    Config["vocab_size"] = bertConfig.vocab_size
    # 1.创建模型
    model = SequenceLabelModel(Config)
    # 2.加载权重
    model_path = os.path.join(model_base_path, Config["model_type"] + ".pth")
    model.load_state_dict(torch.load(model_path))
    return model

def match_entity(sentences, pred_labels):
    assert len(sentences) == len(pred_labels)
    new_sentences = []
    Schema = {v: k for k, v in Labels.items()}
    for text, labels in zip(sentences, pred_labels):
        labels = labels[: len(text)]
        new_text = ""
        for char, label in zip(text, labels):
            # 给每个字符加上预测标签
            new_text += char + "(" + Schema[int(label)] + ")"
        new_sentences.append(new_text)
    return new_sentences


def predict(sentences):
    model = load_model(Config["model_base_path"])
    vocab_dict = load_vocab(Config["vocab_path"])
    sequences = []
    for text in sentences:
        seq = sentence2sequence(text, vocab_dict, Config["max_length"])
        sequences.append(seq)
    sequences = torch.LongTensor(sequences)
    pred_labels = model(sequences)
    return match_entity(sentences, pred_labels)



if __name__ == "__main__":
    texts = ["马丁路德金在演讲台上说：‘今天我有一个梦想...’",
             "2001年9月11日，美国金融大厦发生恐怖袭击",
             "2008年奥巴马称为美国历史上第一位非裔总统",
             "9月3日那天，习近平在北京天安门广场观看了盛大的阅兵仪式"]
    new_text = predict(texts)
    for s in new_text:
        print(s)