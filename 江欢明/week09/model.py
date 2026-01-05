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
        max_length = config["max_length"]
        class_num = config["class_num"]

        # 创建BERT模型
        self.bert_config = BertConfig.from_pretrained(config["bert_path"])
        self.bert = BertModel(self.bert_config)

        # 需要将字符ID映射到BERT的embedding
        # 获取BERT的word embedding层，但我们会替换它
        self.embedding = nn.Embedding(vocab_size, self.bert_config.hidden_size, padding_idx=0)

        # 覆盖BERT原本的word embeddings
        self.bert.embeddings.word_embeddings = self.embedding

        # 分类层
        self.classify = nn.Linear(self.bert_config.hidden_size, class_num)

        # CRF层
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]

        # 损失函数
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        # 创建attention mask，0表示padding位置
        attention_mask = (x != 0).long()

        # 通过BERT获取序列表示
        bert_outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 取最后一层的输出 [batch_size, seq_len, hidden_size]
        sequence_output = bert_outputs.last_hidden_state

        # 分类层
        predict = self.classify(sequence_output)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)  # 忽略padding位置
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
        # 对BERT参数和其他参数使用不同的学习率
        bert_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        optimizer = Adam([
            {'params': bert_params, 'lr': learning_rate},
            {'params': other_params, 'lr': learning_rate * 10}
        ])
    elif optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate)

    return optimizer


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
