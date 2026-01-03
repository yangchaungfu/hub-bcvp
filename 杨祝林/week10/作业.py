# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertConfig

"""
BERT自回归文本生成（终极修复版，解决形状不匹配问题）
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.input_dim = input_dim
        self.window_size = 10

        # 轻量化BERT配置
        self.config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=input_dim,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=input_dim * 2,
            pad_token_id=0,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )

        # 加载BERT并替换注意力层为单向
        self.bert = BertModel(self.config)
        # 手动修改BERT的注意力为单向（核心修复）
        for layer in self.bert.encoder.layer:
            layer.attention.self.is_causal = True  # 开启因果掩码

        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.2)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        batch_size, seq_len = x.shape

        # 仅传2D的padding mask（符合BERT要求）
        attention_mask = (x != 0).long()  # [batch_size, seq_len]

        # BERT前向（仅传padding mask，因果掩码已内置）
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )

        x = outputs.last_hidden_state
        x = self.dropout(x)
        y_pred = self.classify(x)

        if y is not None:
            return self.loss(y_pred.view(-1, self.vocab_size), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 词表/数据加载逻辑（无修改）
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]
            vocab[char] = index + 1
    if "[UNK]" not in vocab:
        vocab["[UNK]"] = len(vocab)
    return vocab


def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]
    x = [vocab.get(word, vocab["[UNK]"]) for word in window]
    y = [vocab.get(word, vocab["[UNK]"]) for word in target]
    return x, y


def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 生成逻辑（无修改）
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    stop_chars = ["，", "。", "、", "的", "了", "在", "上", "下"]
    last_3_chars = []

    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30:
            # 输入长度对齐
            input_text = openings[-window_size:] if len(openings) >= window_size else openings
            x = [vocab.get(char, vocab["[UNK]"]) for char in input_text]
            while len(x) < window_size:
                x.insert(0, vocab["<pad>"])
            x = torch.LongTensor([x])

            if torch.cuda.is_available():
                x = x.cuda()

            # 预测下一个字符
            y = model(x)[0][-1]
            index = sampling_strategy(y, vocab, stop_chars, last_3_chars)
            pred_char = reverse_vocab.get(index, "[UNK]")

            # 跳过无效字符
            if pred_char in ["<pad>", "[UNK]"]:
                pred_char = ""
                continue

            # 防止重复
            last_3_chars.append(pred_char)
            if len(last_3_chars) > 3:
                last_3_chars.pop(0)

            openings += pred_char
    return openings


# 采样策略（无修改）
def sampling_strategy(prob_distribution, vocab, stop_chars, last_3_chars):
    # 降低重复字符概率
    for char in last_3_chars + stop_chars:
        if char in vocab:
            char_id = vocab[char]
            prob_distribution[char_id] *= 0.1

    # 采样逻辑
    if random.random() > 0.15:
        index = int(torch.argmax(prob_distribution))
    else:
        prob_np = prob_distribution.cpu().numpy()
        threshold = np.mean(prob_np)
        valid_indices = np.where(prob_np > threshold)[0]
        if len(valid_indices) == 0:
            valid_indices = list(range(len(prob_np)))
        valid_probs = prob_np[valid_indices] / sum(prob_np[valid_indices])
        index = np.random.choice(valid_indices, p=valid_probs)

    return index


def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for word in window]
            while len(x) < window_size:
                x.insert(0, vocab["<pad>"])
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


# 训练函数（无修改）
def train(corpus_path, save_weight=True):
    epoch_num = 15
    batch_size = 64
    train_sample = 30000
    char_dim = 256
    window_size = 10
    vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    model = build_model(vocab, char_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    # 优化器配置
    optim = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)

    print("BERT自回归模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            watch_loss.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(watch_loss)
        print("=========")
        print(f"第{epoch + 1}轮平均loss:{avg_loss:.6f} | 当前学习率:{scheduler.get_last_lr()[0]:.6f}")
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))

    if save_weight:
        os.makedirs("model", exist_ok=True)
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("corpus.txt", False)
