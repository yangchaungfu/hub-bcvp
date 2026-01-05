# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.use_crf = config["use_crf"]
        class_num = config["class_num"]

        # BERT隐藏层维度
        bert_hidden_size = self.bert.config.hidden_size
        # 分类层，将BERT输出映射到标签空间
        self.classify = nn.Linear(bert_hidden_size, class_num)
        # CRF层
        self.crf_layer = CRF(class_num, batch_first=True)
        # 损失函数
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, target=None):
        # x: 输入文本的token ids，shape: (batch_size, sen_len)
        # attention_mask: 用于区分真实token和填充token，shape: (batch_size, sen_len)

        # BERT输出: last_hidden_state shape: (batch_size, sen_len, bert_hidden_size)
        bert_output = self.bert(input_ids=x, attention_mask=attention_mask)[0]
        predict = self.classify(bert_output) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                # 关键1：替换target中的-100为合法标签（0=O标签）
                tags = torch.where(
                    target == -100,
                    torch.tensor(0, device=target.device),  # 替换值需与设备一致
                    target
                )
                # 关键2：生成有效mask（过滤-100和padding）
                mask = (target != -100) & attention_mask.bool()
                # 关键3：强制第一个时间步mask为1（CRF要求）
                mask[:, 0] = True
                # 关键4：过滤无有效token的样本
                valid_sample_mask = mask.any(dim=1)
                if not valid_sample_mask.all():
                    # 仅保留有效样本（避免CRF计算失败）
                    predict = predict[valid_sample_mask]
                    tags = tags[valid_sample_mask]
                    mask = mask[valid_sample_mask]
                    if len(predict) == 0:
                        raise ValueError("批次中无有效样本，请检查数据！")
                # 计算CRF损失（负对数似然取反）
                loss = -self.crf_layer(predict, tags, mask=mask, reduction="mean")
                return loss
            else:
                # 交叉熵损失（自动忽略-100）
                return self.loss(predict.reshape(-1, self.class_num), target.reshape(-1))
        else:
            # 推理阶段
            if self.use_crf:
                # 核心修复：处理 attention_mask 为 None 的情况 + 保证 mask 维度/设备对齐
                if attention_mask is None:
                    # 场景1：无 attention_mask 时，生成全 True 的 mask（默认所有位置有效）
                    # 注意：predict 是 CRF 输出的对数概率，维度通常为 [batch_size, seq_len, num_tags]
                    batch_size, seq_len = predict.shape[:2]
                    infer_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=predict.device)
                else:
                    # 场景2：有 attention_mask 时，转换为 bool 类型（兼容整型 mask：1=有效，0=无效）
                    infer_mask = attention_mask.bool()
                infer_mask[:, 0] = True
                return self.crf_layer.decode(predict, mask=infer_mask)
            else:
                return torch.argmax(predict, dim=-1)  # 推理返回标签ID


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