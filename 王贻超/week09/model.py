# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

"""
建立基于BERT的网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]
        self.class_num = config["class_num"]
        
    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 兼容不同版本的transformers
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state  # 新版本
        else:
            sequence_output = outputs[0]  # 旧版本，第一个元素是last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, class_num)
        
        if labels is not None:
            if self.use_crf:
                # 确保mask是布尔类型，并且第一个时间步为True
                if attention_mask is not None:
                    mask = attention_mask.bool()
                    # 强制设置第一个时间步为True，满足CRF要求
                    mask[:, 0] = True
                else:
                    # 如果没有attention_mask，则创建一个全为True的mask
                    mask = torch.ones_like(labels, dtype=torch.bool)
                return - self.crf_layer(logits, labels, mask, reduction="mean")
            else:
                # 计算交叉熵损失，忽略标签为-1的位置
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.class_num)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                return loss_fct(active_logits, active_labels)
        else:
            if self.use_crf:
                if attention_mask is not None:
                    mask = attention_mask.bool()
                    # 同样确保第一个时间步为True
                    mask[:, 0] = True
                else:
                    mask = None
                return self.crf_layer.decode(logits, mask)
            else:
                return torch.argmax(logits, dim=-1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        # 对BERT参数和分类器参数使用不同的学习率
        bert_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0].startswith('bert'), model.named_parameters()))))
        classifier_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0].startswith('classifier') or kv[0].startswith('crf'), model.named_parameters()))))
        optimizer = Adam([
            {'params': bert_params, 'lr': learning_rate * 0.1},  # BERT参数使用较小的学习率
            {'params': classifier_params, 'lr': learning_rate}
        ])
        return optimizer
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
