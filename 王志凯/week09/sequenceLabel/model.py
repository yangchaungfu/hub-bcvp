# -*- coding:utf-8 -*-


import os
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from logHandler import logger
log = logger(os.path.basename(__file__))

class SequenceLabelModel(nn.Module):
    def __init__(self, config):
        super(SequenceLabelModel, self).__init__()
        # 参数
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.model_type = config["model_type"]
        self.label_num = config["label_num"]
        self.use_crf = config["use_crf"]
        self.config = config
        # 网络层
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        if self.model_type == "lstm":
            self.encoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
            self.hidden_size *= 2
        elif self.model_type == "bert":
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=True)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.label_num)
        self.crf = CRF(self.label_num, batch_first=True)

    def forward(self, x, tags=None):
        if self.model_type == "lstm":
            x = self.embedding(x)
            x, _ = self.encoder(x)
        elif self.model_type == "bert":
            x = self.encoder(x).last_hidden_state
        else:
            x = self.encoder(x)
        emissions = self.linear(x)
        log.info(f"emissions.shape:{emissions.shape}")  # (batch_size, seq_len, label_num)
        # tags为空，进行预测
        if tags is None:
            if self.use_crf:
                # 使用crf.decode解码，返回batch中每个样本的最优的标签序列列表（batch_size, seq_len）
                return self.crf.decode(emissions)
            return torch.softmax(emissions, dim=-1)
        # 传入tags，计算loss
        else:
            if self.use_crf:
                # 使用crf计算loss时，传入预测值和真实值（mask可选），最后一定要加负号
                mask = tags != -1
                return -self.crf(emissions, tags, mask)
            else:
                loss = nn.CrossEntropyLoss(ignore_index=-1)
                # 序列标注任务中，预测值(batch_size, seq_len, label_num)不能与真实标签(batch_size, seq_len)直接计算loss
                # 需要转化成：预测值（batch_size * seq_len, label_num），真实标签（batch_size * seq_len）
                emissions = emissions.view(-1, emissions.shape[-1])
                tags = tags.veiw(-1)   # 等效于 tags.flatten()，但是效率比view低
                log.info(f"emissions.shape:{emissions.shape}，tags.shape:{tags.shape}")
                return loss(emissions, tags)




