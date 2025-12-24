# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random

from transformers import BertTokenizer, BertModel

"""
采用BERT模型完成文本生成任务
"""

BASE_MODEL_PATH = r"D:\Miniconda3\bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_PATH)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(BASE_MODEL_PATH)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask, target=None):
        """
        Args:
            input_ids: shape: [batch_size, sen_len]
            attention_mask: shape: [batch_size, sen_len, sen_len]
            target: shape: [batch_size, sen_len]
        """
        # print(">> input_ids:", input_ids)
        # print(">> attention_mask:", attention_mask)
        # print(">> target:", target)
        # [batch_size, sen_len] -> [batch_size, sen_len, input_dim]
        sequence_out, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        # [batch_size, sen_len, input_dim] -> [batch_size, sen_len, vocab_size]
        logits = self.classify(sequence_out)
        if target is not None:
            loss = self.loss(logits.view(-1, logits.shape[-1]), target.view(-1))
            print(">> loss: ", loss)
            return loss
        else:
            return torch.softmax(logits, dim=-1)


def load_corpus(path):
    corpus = ""
    with open(path, encoding="utf8") as f:
        for line in f:
            corpus += line.strip()
            corpus += "\n"
    return corpus


# 生成一个样本：从语料中随机截取字符串s[i:j]，s[i:j]作为模型输入，以s[i+1:j+1]作为预测目标
def build_sample(corpus, window_size):
    i = random.randint(0, len(corpus) - 1 - window_size)
    j = i + window_size
    input_text = corpus[i:j]
    target_text = corpus[i + 1:j + 1]  # 输入输出错开一位

    input_ids = encode_sentence(input_text, window_size)
    target = encode_sentence(target_text, window_size)
    # 用下三角矩阵做掩码，使得bert在处理第i个token时只有第i个token左侧的token是可见的
    # 原理：attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
    attention_mask = np.tril(np.ones((window_size, window_size), dtype=np.int32)).tolist()
    return input_ids, attention_mask, target


# 建立数据集
def build_dataset(sample_num, corpus, sentence_len):
    ds_input_ids = []
    ds_atten_mask = []
    ds_target = []
    for i in range(sample_num):
        input_ids, atte_mask, target = build_sample(corpus, sentence_len)
        ds_input_ids.append(input_ids)
        ds_atten_mask.append(atte_mask)
        ds_target.append(target)
    return torch.LongTensor(ds_input_ids), torch.LongTensor(ds_atten_mask), torch.LongTensor(ds_target)


def encode_sentence(sentence, sentence_len, truncation=True):
    input_ids = tokenizer.encode(
        sentence
        , max_length=sentence_len
        , padding="max_length"
        , truncation=truncation
        , add_special_tokens=False
        , return_tensors=None  # "pt": 返回tensor, "None": 返回list
    )
    return input_ids


# 文本生成测试代码
def generate_sentence(openings, sentence_len, model):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 如果生成了换行符 或 生成的文本超过限制， 则终止迭代
        while pred_char != "\n" and len(openings) <= 70:
            openings += pred_char
            # 从openings的尾部截取长度为sentence_len的字符串
            input_ids = encode_sentence(openings[-sentence_len:], sentence_len, truncation=False)
            input_ids = torch.LongTensor([input_ids])
            attention_mask = torch.ones((len(input_ids), len(input_ids)), device=input_ids.device).unsqueeze(0)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            prob_distribution = model(input_ids, attention_mask)[0][-1]  # 预测的最后一个token的概率分布

            # 根据采样策略选择预测的字，但如果选择的字是特殊token，则重新采样
            pred_char = ""
            while pred_char == "" or pred_char in tokenizer.special_tokens_map.values():
                index = sampling_strategy(prob_distribution)
                pred_char = tokenizer.decode([index])
            # print(">>", index, pred_char)
    return openings


# 采样策略：90%取概率最大的token，10%的概率从topK中随机选择
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        return int(torch.argmax(prob_distribution))
    else:
        # 从概率最大的K个token中随机选取
        _, idxes = torch.topk(prob_distribution, k=30)
        return random.choice(idxes.tolist())


def train(corpus_path, save_weight=True):
    epoch_num = 16  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample_num = 40000  # 每轮训练总共训练的样本总数
    learning_rate = 0.001  # 学习率

    sentence_len = 40  # 样本文本长度
    embed_size = 768  # bert默认词向量维度

    corpus = load_corpus(corpus_path)  # 加载语料
    model = LanguageModel(len(tokenizer.vocab), embed_size)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器

    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample_num / batch_size)):
            input_ids, atten_mask, target = build_dataset(batch_size, corpus, sentence_len)  # 构建一组训练样本
            if torch.cuda.is_available():
                input_ids, atten_mask, target = input_ids.cuda(), atten_mask.cuda(), target.cuda()

            optim.zero_grad()  # 梯度归零
            loss = model(input_ids, atten_mask, target)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮 的平均loss: %f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", sentence_len, model))
        print(generate_sentence("李慕站在山路上，深深的呼吸", sentence_len, model))

    if save_weight:
        torch.save(model.state_dict(), "text_generation_by_bert.py")
        return


if __name__ == "__main__":
    train("corpus.txt", False)
