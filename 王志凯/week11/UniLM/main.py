# -*- coding:utf-8 -*-

"""
使用UniLM的seq-to-seq mask进行提取摘要任务训练
"""
import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import *
from transformers import BertModel, BertConfig, BertTokenizer

bertConfig = BertConfig.from_pretrained(BERT_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
cuda = torch.cuda.is_available()


class UniLMSeq2SeqModel(nn.Module):
    def __init__(self):
        super(UniLMSeq2SeqModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH, return_dict=True)
        self.linear = nn.Linear(bertConfig.hidden_size, bertConfig.vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, target=None, mask=None):
        output = self.bert(inputs, attention_mask=mask).last_hidden_state
        # [batch_size, seq_len, v_size]
        pred = self.linear(output)
        if target is not None:
            # 由于输入序列是将源序列和目标序列拼接而成，所以pred和target不等长，源序列部分不需要计算loss，故将pred中的源序列部分截断
            # 输入：[cls, 1, 2, 3, sep, 4, 5, sep] -> 截断：[4, 5, sep]
            pred = pred[:, INPUT_MAX_LENGTH:, :]
            # pred:[batch_size, seq_len, v_size] -> [batch_size × seq_len, v_size]
            # target:[batch_size, seq_len] -> [batch_size × seq_len]
            return self.loss(pred.contiguous().view(-1, pred.size(-1)), target.view(-1))
        else:
            return torch.softmax(pred, dim=-1)

class UniLMDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
        self.load()

    def load(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    title = data["title"]
                    content = data["content"]
                    title_seq = tokenizer.encode(title, padding='max_length', truncation=True,
                                                 max_length=TARGET_MAX_LENGTH)
                    content_seq = tokenizer.encode(content, padding='max_length', truncation=True,
                                                   max_length=INPUT_MAX_LENGTH)
                    # 将前面的cls去掉作为目标序列，用于计算loss
                    target = title_seq[1:]
                    # 将content作为输入，title作为输出，进行拼接
                    input_concat = torch.cat([torch.LongTensor(content_seq), torch.LongTensor(target)], dim=-1)
                    self.data.append([input_concat, torch.LongTensor(target)])
            print(f"训练数据加载完成！")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_mask(s1_len, s2_len):
    # 当对源序列和目标序列进行拼接时，去掉了中间的一个cls，所以mask的seq_len长度也要减一
    seq_len = s1_len + s2_len - 1
    """
    第一步：先创建全零mask
                    s2
    s1 [0, 0, 0, 0, 0, 0, 0]
       [0, 0, 0, 0, 0, 0, 0]
       [0, 0, 0, 0, 0, 0, 0]
       [0, 0, 0, 0, 0, 0, 0]
    s2 [0, 0, 0, 0, 0, 0, 0]
       [0, 0, 0, 0, 0, 0, 0]
       [0, 0, 0, 0, 0, 0, 0]
    """
    mask = torch.zeros(seq_len, seq_len)
    """
    第二步：将左半区置为1
                    s2
    s1 [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
    s2 [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
    """
    mask[:, :s1_len] = 1
    """
    第三步：将右下角置为下三角mask
                    s2
    s1 [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
       [1, 1, 1, 1, 0, 0, 0]
    s2 [1, 1, 1, 1, 1, 0, 0]
       [1, 1, 1, 1, 1, 1, 0]
       [1, 1, 1, 1, 1, 1, 1]
    """
    for index, i in enumerate(range(s1_len, seq_len)):
        # 将第i列的s1_len+index行置为1
        mask[s1_len+index:, i] = 1
    # 添加batch维度，和输入维度保持一致
    return mask.contiguous().view(1, seq_len, seq_len)


def main():
    model = UniLMSeq2SeqModel()
    if cuda:
        model = model.cuda()
    train_data = DataLoader(UniLMDataset(CORPUS_PATH), batch_size=BATCH_SIZE, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 生成UniLM Seq-to-Seq mask
    mask = load_mask(INPUT_MAX_LENGTH, TARGET_MAX_LENGTH)

    model.train()
    for epoch in range(EPOCHS):
        print(f"===开始第{epoch+1}轮训练===")
        watch_loss = []
        count = 0
        for i, batch_data in enumerate(train_data):
            if cuda:
                batch_data = [b.cuda() for b in batch_data]
            batch_x, batch_y = batch_data
            loss = model(batch_x, batch_y, mask)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
            # 打印训练进度（每10%打印一次）
            if i % int(len(train_data) / 10) == 0 and i > 0:
                count += 1
                print(f"训练进度：{count / 10:.0%}")
        print(f"该轮训练结束，平均loss：{np.mean(watch_loss):.5f}")
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "model.pth")
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
    # mask = load_mask(4, 3)
    # print(mask)
