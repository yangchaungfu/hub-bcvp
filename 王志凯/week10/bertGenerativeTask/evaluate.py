# -*- coding:utf-8 -*-

import torch
import random
import math
import numpy as np
from loader import sentence2sequence, load_vocab
from logHandler import logger
log = logger(__file__)

class Evaluator:
    def __init__(self, config, model, sentence):
        self.model = model
        self.config = config
        self.sentence = sentence
        self.vocab = load_vocab(config["vocab_path"])


    def eval(self):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.model.eval()
        with torch.no_grad():
            pred_char = ""
            probs = 0
            count = 0
            while len(self.sentence) <= 40 and pred_char != "\n":
                sequence = sentence2sequence(self.sentence, self.vocab, len(self.sentence))
                x = torch.LongTensor([sequence])
                log.info(f"sentence:{self.sentence}, sequence:{x}")
                # 取出第一个batch中的最后一个隐单元，即为预测词的概率分布
                y_pred = self.model(x)[0][-1]
                log.info(f"y_pred.shape:{y_pred.shape}")
                # 取出原句中的最后一个字的概率值（不是预测字）作为P(w_i|w_1...w_i-1)
                prob = calc_ppl(sequence, y_pred)
                probs += prob
                # 根据不同的采样策略选择预测词
                pred_id = sampling_strategy(y_pred)
                log.info(f"pred_id:{pred_id}")
                pred_char = reverse_vocab[pred_id]
                # 追加到原句子中继续生成
                self.sentence += pred_char
                count += 1
            log.info(f"最终生成的句子：{self.sentence}")
            print(f"最终生成的句子：{self.sentence}")
        # 计算ppl（这里计算的是生成的句子成句概率，并不包含原句，所以不应该除以len(sequence)）
        ppl = 2 ** (-probs / count)
        log.info(f"生成的句子PPL值为：{ppl}")
        print(f"生成的句子PPL值为：{ppl}")

def calc_ppl(sequence, y_pred):
    target = sequence[-1]
    target_prob = y_pred[target]
    return math.log(target_prob, 10)
    # return math.exp(target_prob)


def sampling_strategy(y_pred):
    margin = random.random()
    # 20%概率为贪婪采样，80%为随机采样
    if margin > 0.2:
        return int(torch.argmax(y_pred, dim=-1))
    else:
        # 确保归一化（内部处理精度问题）
        y_pred = y_pred.cpu().numpy()
        # 按照y_pred中各元素的概率进行采样
        return np.random.choice(list(range(len(y_pred))), p=y_pred)