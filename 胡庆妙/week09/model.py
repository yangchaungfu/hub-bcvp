# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF  # pip install pytorch-crf
from transformers import BertModel

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        embed_size = config["embed_size"]
        class_num = config["class_num"]

        self.bert = BertModel.from_pretrained(config["bert_model_path"])
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(embed_size, class_num)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, target=None):
        """
        Args:
            input_ids: shape: [batch_size, sen_len]
            target:  shape: [batch_size, sen_len]
        """
        attention_mask = (input_ids != 0).long()  # 假设非0都是有效token

        # 调用 bert， [batch_size, sen_len -> [batch_size, sen_len, embed_size]
        sequence_output, _ = self.bert(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       return_dict=False)
        predict = self.classify(sequence_output)  # shape: -> [batch_size, sen_len, class_num]
        if target is not None:
            # predict变形为[batch_size * sen_len, class_num], target变形为[batch_size * sen_len]
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            # [batch_size, sen_len, class_num] -> [batch_size, sen_len]
            return torch.argmax(predict, dim=-1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
