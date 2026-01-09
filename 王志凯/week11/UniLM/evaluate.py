# -*- coding:utf-8 -*-

import os
import torch
import json
import random
import numpy as np
from config import *
from main import UniLMSeq2SeqModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
cuda = torch.cuda.is_available()


def load_test_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(line.strip())
    return data


def sentence2sequence(test_data):
    sequences = tokenizer(test_data, padding="max_length", truncation=True, max_length=INPUT_MAX_LENGTH,
                          return_tensors="pt")["input_ids"]
    print(f"sequences.shape:{sequences.shape}")  # [10, seq_len]
    return sequences


def evaluate(model_path, test_data):
    # 加载模型和数据
    model = UniLMSeq2SeqModel()
    model.load_state_dict(torch.load(model_path))
    if cuda:
        model = model.cuda()
    input_ids = sentence2sequence(test_data)

    model.eval()
    with torch.no_grad():
        pred = model(input_ids)  # [batch_size, seq_len, v_size]
        print(f"pred.shape:{pred.shape}")

        # 采样策略
        pred_ids = torch.argmax(pred, dim=-1)
        print(f"pred_ids.shape:{pred_ids.shape}")  # [batch_size, seq_len]

        # 将序列转化为文本
        sentences = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        sentences = [sent.replace(' ', '') for sent in sentences]
        assert len(test_data) == len(sentences)
        for data, sentence in zip(test_data, sentences):
            print(f"输入文本：{data}")
            print(f"回答：{sentence}")
            print("=" * 50)


if __name__ == '__main__':
    model_path = os.path.join(MODEL_DIR, "model.pth")
    test_data = load_test_data(TEST_PATH)
    evaluate(model_path, test_data)
