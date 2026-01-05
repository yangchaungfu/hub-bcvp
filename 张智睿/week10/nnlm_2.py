#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
import re
from transformers import BertModel, T5Model
import transformers

"""
基于pytorch的BERT(Encoder) + MLM训练
- 将原LSTM next-token训练改为 BERT masked LM 训练
- 训练中加入mask，labels仅在mask位置监督，避免“作弊式抄输入”
"""

IGNORE_INDEX = -100


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]
        self.mask_id = vocab["<MASK>"]

        vocab_size = len(vocab)

        # 使用 BertModel + 自建MLM head
        num_heads = 8
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim({input_dim}) 必须能被 num_heads({num_heads}) 整除")

        config = transformers.BertConfig(
            vocab_size=vocab_size,
            hidden_size=input_dim,
            num_hidden_layers=4,
            num_attention_heads=num_heads,
            intermediate_size=input_dim * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=self.pad_id,
        )

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        """
        x: [B, T] input_ids（已经做过mask的输入）
        y: [B, T] labels（只有mask位置为真实id，其它位置为 -100）
        """
        attention_mask = (x != self.pad_id).long()
        token_type_ids = torch.zeros_like(x)

        out = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        h = out.last_hidden_state  # [B, T, H]
        h = self.dropout(h)
        y_pred = self.classify(h)  # [B, T, V]

        if y is not None:
            return self.loss_fn(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表，自动补 <MASK>
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]        # 去掉结尾换行符
            vocab[char] = index + 1 # 留出0位给pad token

    if "<UNK>" not in vocab:
        raise ValueError("vocab.txt 必须包含 <UNK>。")

    # MLM必须有mask token；若vocab里没有就自动加
    if "<MASK>" not in vocab:
        vocab["<MASK>"] = len(vocab)

    return vocab


# 加载语料（GBK）
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def apply_mlm_mask(token_ids, vocab_size, mask_id, pad_id=0, mask_prob=0.15):
    """
    MLM mask策略（简化版，BERT经典80/10/10）：
    - 随机选mask_prob比例的位置作为监督目标
    - 被选中的位置：
        80% -> 替换为 <MASK>
        10% -> 替换为随机token
        10% -> 保持原token
    labels：
    - mask位置：原token id
    - 非mask位置：-100（ignore，不计入loss）
    """
    labels = [IGNORE_INDEX] * len(token_ids)
    masked = list(token_ids)

    # 候选位置：排除pad与mask自身
    candidate = [i for i, t in enumerate(token_ids) if t != pad_id and t != mask_id]
    if not candidate:
        return masked, labels

    masked_pos = [i for i in candidate if random.random() < mask_prob]

    # 至少mask一个位置，否则loss可能变得“无监督”
    if len(masked_pos) == 0:
        masked_pos = [random.choice(candidate)]

    for i in masked_pos:
        original = token_ids[i]
        labels[i] = original

        r = random.random()
        if r < 0.8:
            masked[i] = mask_id
        elif r < 0.9:
            rid = random.randint(1, vocab_size - 1)
            if rid == mask_id:
                rid = 1
            masked[i] = rid
        else:
            masked[i] = original

    return masked, labels


# 随机生成一个样本（改为MLM：返回 masked_x 和 labels）
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]

    x = [vocab.get(ch, vocab["<UNK>"]) for ch in window]
    x_masked, y_labels = apply_mlm_mask(
        token_ids=x,
        vocab_size=len(vocab),
        mask_id=vocab["<MASK>"],
        pad_id=vocab["<pad>"],
        mask_prob=0.15
    )
    return x_masked, y_labels


# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码（用“上下文 + <MASK>”预测下一个字）
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30:
            # 取 window_size-1 个上下文，最后补一个 <MASK> 预测下一个字符
            if window_size <= 1:
                context = ""
            else:
                context = openings[-(window_size - 1):]

            x_ids = [vocab.get(ch, vocab["<UNK>"]) for ch in context] + [vocab["<MASK>"]]
            x = torch.LongTensor([x_ids])
            if torch.cuda.is_available():
                x = x.cuda()

            # 取最后一个位置（<MASK>）的预测分布
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab.get(index, "<UNK>")
            openings += pred_char

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


def train(corpus_path, save_weight=True):
    epoch_num = 20         # 训练轮数
    batch_size = 64        # 每次训练样本个数
    train_sample = 50000   # 每轮训练总共训练的样本总数
    char_dim = 256         # hidden_size
    window_size = 32       # 样本文本长度

    vocab = build_vocab("vocab.txt")       # 建立字表（自动补<MASK>）
    corpus = load_corpus(corpus_path)      # 加载语料
    model = build_model(vocab, char_dim)   # 建立模型

    if torch.cuda.is_available():
        model = model.cuda()

    # BERT/Transformer 学习率用1e-4
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("文本词表模型加载完毕（BERT + MLM mask训练），开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本（已mask）
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()
            loss = model(x, y)     # 计算loss（只在mask位置）
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("corpus.txt", False)
