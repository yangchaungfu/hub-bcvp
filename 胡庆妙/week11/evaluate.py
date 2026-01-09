# -*- coding: utf-8 -*-
import random
import sys

import torch
from transformers import BertTokenizer

import loader
from collections import defaultdict

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.logger = logger
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.tokenizer.vocab[" "] = len(self.tokenizer.vocab)  # 在词表中增加空格符
        self.tokenizer.vocab["[EOS]"] = len(self.tokenizer.vocab)  # 在词表中语句结束标识符

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()

        self.valid_data = loader.load_valid_data(config["valid_data_path"], config, batch_size=1)
        self.stats_dict = defaultdict(int)  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始验证 第%d轮 的训练效果：" % epoch)
        self.stats_dict = defaultdict(int)

        for index, batch_data in enumerate(self.valid_data):
            ask, ans = batch_data  # 每批次一条
            ask = ask[0]
            ans = ans[0]
            pred_ans = generate_answer(ask, sys.maxsize, self.config["output_max_length"], self.tokenizer, self.model)
            print("样本输入：", ask)
            print("样本输出：", ans)
            print("预测输出：", pred_ans)
            break


def generate_answer(ask, input_max_len, output_max_len, tokenizer: BertTokenizer, model):
    """
    Args:
        ask: 问答系统的问句
        input_max_len: 输出序列的最大长度
        output_max_len: 输出序列的最大长度
        tokenizer: 分词器
        model: 模型
    Returns:
        第二条语句，或是问答系统的答句
    """
    ask_seq = [tokenizer.cls_token_id] + tokenizer.encode(ask, add_special_tokens=False)
    output_text = ""
    with torch.no_grad():
        cnt = 0
        pred_word = tokenizer.sep_token  # decoder的第一个token

        # 如果生成的字符中出现换行、[EOS]，则生成结束
        while pred_word != "\n" and pred_word != "[EOS]" and cnt <= output_max_len:
            ask_seq.append(tokenizer.vocab[pred_word])  # 将预测的词添加到输入序列中
            input_ids = ask_seq  # 控制输入序列的最大长度, 取尾部的max_len个序列
            if len(input_ids) > input_max_len:
                input_ids = ask_seq[-1 * input_max_len:]

            input_ids = torch.LongTensor([input_ids])
            attention_mask = torch.ones((len(input_ids), len(input_ids)), device=input_ids.device).unsqueeze(0)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda
            logits = model(input_ids, attention_mask)[0][-1]  # 最后一个token的未规范化的预测值

            # 根据采样策略选择预测的字
            vocab_idx = sampling_strategy(logits)
            pred_word = tokenizer.decode([vocab_idx])
            output_text += pred_word
            cnt += 1
            # print(">>", cnt, pred_word)
    return output_text.replace(tokenizer.pad_token, "")


def sampling_strategy(logits, temperature=0.8):
    prob_distribution = torch.softmax(logits / temperature, dim=0)
    # print("概率分布：", prob_distribution)

    if random.random() > 0.05:
        return int(torch.argmax(prob_distribution))
    else:
        # 从概率最大的K个token中随机选取
        _, idxes = torch.topk(prob_distribution, k=3)
        return random.choice(idxes.tolist())
