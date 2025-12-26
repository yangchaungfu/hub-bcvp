# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertConfig
from causalBert import causalBertModel
from logHandler import logger
log = logger(__file__)


class BertGenerativeModel(nn.Module):
    def __init__(self, config):
        super(BertGenerativeModel, self).__init__()
        self.vocab_size = config["vocab_size"]
        self.hidden_size = BertConfig.from_pretrained(config["bert_path"]).hidden_size
        self.bert = causalBertModel(config)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, target=None):
        output = self.bert(input_ids)
        x = output.last_hidden_state    # shape:(batch_size, seq_len, char_dim)
        y_pred = self.linear(x)         # shape:(batch_size, seq_len, vocab_len)
        if target is not None:
            # y_pred:(batch_size, seq_len, vocab_len)  -> (batch_size * seq_len, vocab_len)
            # target:(batch_size, seq_len)  ->  (batch_size * seq_len,)
            y_pred = y_pred.view(-1, y_pred.size(-1))
            y_true = target.view(-1)
            return self.loss(y_pred, y_true)
        else:
            return torch.softmax(y_pred, dim=-1)