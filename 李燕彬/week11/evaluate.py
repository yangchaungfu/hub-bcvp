# -*- coding: utf-8 -*-
import torch
import numpy as np
import random


class Evaluator:
    def __init__(self, config, model, text_length, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.text_length = text_length


    def eval(self):
        pass

    #生成content内容
    def generate_content(self, title, model, vocab, max_input_len, max_output_len):
        """
        根据输入的title生成对应的content
        
        参数:
            title: 输入的标题文本
            model: 训练好的Seq2Seq模型
            vocab: 词表
            max_input_len: 输入序列的最大长度
            max_output_len: 输出序列的最大长度
        返回:
            生成的content文本
        """
        reverse_vocab = dict((y, x) for x, y in vocab.items())
        model.eval()
        
        with torch.no_grad():
            # 处理输入标题
            input_ids = [vocab.get(char, vocab["<UNK>"]) for char in title[:max_input_len]]
            
            # 填充到最大长度
            if len(input_ids) < max_input_len:
                input_ids += [vocab["<pad>"]] * (max_input_len - len(input_ids))
            
            input_tensor = torch.LongTensor([input_ids])
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # 生成content
            output_probs = model(input_tensor)
            
            # 将概率转换为token序列
            output_ids = torch.argmax(output_probs[0], dim=-1).tolist()
            
            # 构建生成的文本
            generated_text = []
            for token_id in output_ids:
                if token_id == vocab["</s>"] or token_id == vocab["<pad>"]:
                    break
                generated_text.append(reverse_vocab.get(token_id, "<UNK>"))
            
            return "".join(generated_text)

    #文本生成测试代码（兼容旧方法）
    def generate_sentence(self, openings, model, vocab, window_size):
        reverse_vocab = dict((y, x) for x, y in vocab.items())
        model.eval()
        with torch.no_grad():
            pred_char = ""
            #生成了换行符，或生成文本超过50字则终止迭代
            while pred_char != "\n" and len(openings) <= self.text_length:
                openings += pred_char
                x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
                x = torch.LongTensor([x])
                if torch.cuda.is_available():
                    x = x.cuda()
                y = model(x)[0][-1]
                index = sampling_strategy(y)
                pred_char = reverse_vocab[index]
        return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

