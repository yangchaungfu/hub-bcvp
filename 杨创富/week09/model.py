# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel  # 需安装transformers库：pip install transformers

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.class_num = config["class_num"]
        self.use_crf = config["use_crf"]
        
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(config["bert_path"])
        # 冻结BERT参数（可选，视情况微调）
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # BERT输出维度为768（base模型），映射到分类数
        self.classifier = nn.Linear(768, self.class_num)
        # 保留CRF层
        self.crf_layer = CRF(self.class_num, batch_first=True)
        # 损失函数（非CRF模式下使用）
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, target=None):
        # BERT输出：last_hidden_state shape=(batch_size, seq_len, 768)
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # last_hidden_state = bert_output.last_hidden_state  # 取最后一层隐藏状态

        last_hidden_state = bert_output[0]  # 原代码是 bert_output.last_hidden_state
    


        
        # 分类头输出
        predict = self.classifier(last_hidden_state)  # shape=(batch_size, seq_len, class_num)
        
        if target is not None:
            if self.use_crf:
                # CRF损失计算（需传入mask忽略padding）
                mask = attention_mask.bool()  # BERT的attention_mask可直接作为CRF的mask
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 交叉熵损失（展平序列维度）
                return self.loss(predict.view(-1, self.class_num), target.view(-1))
        else:
            if self.use_crf:
                # CRF解码
                return self.crf_layer.decode(predict, attention_mask.bool())
            else:
                # 直接取概率最大的标签
                return torch.argmax(predict, dim=-1)


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
