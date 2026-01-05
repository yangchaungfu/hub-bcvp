# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]
        self.bert_layer_index = config.get("bert_layer", 12)  # <-- 保存为实例变量

        # 加载预训练的 BERT 模型，而不是只加载配置
        self.bert = BertModel.from_pretrained(
            config["bert_path"],
            output_hidden_states=True,
            return_dict=True
        )

        # 分类层
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # 字符嵌入
        attention_mask = (x != 0).long()  # 创建 attention mask
        # 直接使用 BERT，它会处理 embedding
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 使用保存的 layer index
        chosen_layer_output = outputs.hidden_states[self.bert_layer_index]  # shape: (batch, seq_len, hidden_size)

        # 通过分类层
        predict = self.classify(chosen_layer_output)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)  # shape: (batch, seq_len)
                # 强制第一个时间步 mask 为 True
                mask[:, 0] = True
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)