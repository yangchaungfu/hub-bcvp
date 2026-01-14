# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator(Dataset):
    def __init__(self, data_path, config, data_type="train"):
        self.path = data_path
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.tokenizer.vocab[" "] = len(self.tokenizer.vocab)  # 在词表中增加空格符
        self.tokenizer.vocab["[EOS]"] = len(self.tokenizer.vocab)  # 在词表中语句结束标识符

        self.data_type = data_type
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                ask = line["content"].strip()  # 样本中的问句
                ans = line["title"].strip()  # 样本中的答句

                if self.data_type == "train":
                    # 针对一对问答样本，构建多条入参；每条入参用于模型的一次调用，每次调用仅预测答句的一个词。
                    train_data = self.build_train_sample(ask, ans, self.config["input_max_length"], self.config["output_max_length"])
                    input_ids_list, attention_mask_list, target_ids_list = train_data
                    for i in range(len(input_ids_list)):
                        self.data.append([input_ids_list[i], attention_mask_list[i], target_ids_list[i]])
                else:
                    self.data.append([ask, ans])
        return

    def build_train_sample(self, ask, ans, ask_max_len, ans_max_len):
        # 问句序列、答句序列
        ask_seq = encode_sentence(self.tokenizer, ask, ask_max_len)
        ans_seq = encode_sentence(self.tokenizer, ans, ans_max_len) + [self.tokenizer.vocab["[EOS]"]]  # [EOS]用于标识语句结束
        combined_seq = ask_seq + [self.tokenizer.sep_token_id] + ans_seq  # [SEP]是分隔符，是decoder的起始token
        sep_idx = len(ask_seq)  # [SEP]的位置
        max_len = len(ask_seq) + 1  # 序列的长度

        input_ids_list = []
        attention_mask_list = []
        target_ids_list = []
        for i in range(len(ask_seq)):  # i指向encoder的第一个token
            if i + max_len >= len(combined_seq):
                break
            encoder_input_ids = combined_seq[i:sep_idx]
            decoder_input_ids = combined_seq[sep_idx:sep_idx + i + 1]

            # 输入序列
            input_ids = torch.tensor(combined_seq[i: i + max_len])
            # 预期输出
            target_ids = torch.tensor(combined_seq[i + 1: i + max_len + 1])

            # 交叉注意力掩码
            encoder_input_mask = torch.ones(len(encoder_input_ids), len(encoder_input_ids), dtype=torch.int32)  # 左上，全1
            decoder_input_mask = torch.zeros(len(encoder_input_ids), len(decoder_input_ids), dtype=torch.int32)  # 右上，全0
            encoder_ouput_mask = torch.ones(len(decoder_input_ids), len(encoder_input_ids), dtype=torch.int32)  # 左下，全1
            decoder_output_mask = torch.tril(torch.ones(len(decoder_input_ids), len(decoder_input_ids), dtype=torch.int32))  # 右下，下三角
            input_mask = torch.hstack([encoder_input_mask, decoder_input_mask])  # 水平拼接
            output_mask = torch.hstack([encoder_ouput_mask, decoder_output_mask])  # 水平拼接
            attention_mask = torch.vstack([input_mask, output_mask])  # 垂直拼接

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            target_ids_list.append(target_ids)

            # print(">>", ask)
            # print(">>", ans)
            # print(">>", input_ids_list)
            # print(">>", attention_mask_list)
            # print(">>", target_ids_list)
        return input_ids_list, attention_mask_list, target_ids_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def encode_sentence(tokenizer, sentence, max_length, padding=True):
    return tokenizer.encode(
        sentence,
        add_special_tokens=False,
        max_length=max_length,
        padding=('max_length' if padding else 'do_not_pad'),
        truncation=True
    )


# 用torch自带的DataLoader类封装数据
def load_train_data(data_path, config):
    dg = DataGenerator(data_path, config, "train")
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=True)
    return dl


# 用torch自带的DataLoader类封装数据
def load_valid_data(data_path, config, batch_size=1):
    dg = DataGenerator(data_path, config, "valid")
    dl = DataLoader(dg, batch_size=batch_size, shuffle=False)
    return dl


if __name__ == "__main__":
    from config import Config

    l_train_data = load_train_data(Config["train_data_path"], Config)
    for index, batch_data in enumerate(l_train_data):
        inpt_ids, atten_mask, trg_ids = batch_data
        print("inputs_ids:", inpt_ids)
        print("attention_mask:", atten_mask)
        print("target_ids:", trg_ids)
        break

    l_valid_data = load_valid_data(Config["valid_data_path"], Config, batch_size=2)
    for index, batch_data in enumerate(l_valid_data):
        ask, ans = batch_data
        print("ask:", ask[0])
        print("ans:", ans[0])
        break
