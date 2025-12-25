# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import AdamW
# from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF


from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW  # 使用torch.optim中的AdamW

class BertNERModel(nn.Module):
    def __init__(self, config):
        super(BertNERModel, self).__init__()
        
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(
            config["bert_path"],
            output_hidden_states=True,
            return_dict=True
        )
        
        # 冻结BERT参数（可选，微调时通常解冻）
        for param in self.bert.parameters():
            param.requires_grad = True  # 微调时设为True
            
        # BERT的隐藏层大小
        bert_hidden_size = self.bert.config.hidden_size
        
        # Dropout层
        self.dropout = nn.Dropout(config["dropout_rate"])
        
        # 分类层（将BERT输出映射到标签空间）
        self.classifier = nn.Linear(bert_hidden_size, config["class_num"])
        
        # CRF层（可选）
        self.use_crf = config["use_crf"]
        if self.use_crf:
            self.crf = CRF(config["class_num"], batch_first=True)
        
        # 损失函数（非CRF时使用）
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
        # 配置参数
        self.config = config
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        前向传播
        Args:
            input_ids: token ids [batch_size, seq_len]
            attention_mask: attention mask [batch_size, seq_len]
            token_type_ids: token type ids [batch_size, seq_len]
            labels: 真实标签 [batch_size, seq_len]
        Returns:
            如果labels不为None: 返回损失
            否则: 返回预测结果
        """
        # 获取BERT输出
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 取最后一层隐藏状态
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 应用dropout
        sequence_output = self.dropout(sequence_output)
        
        # 分类层
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_tags]
        
        if labels is not None:
            # 训练模式：计算损失
            if self.use_crf:
                # CRF损失：需要mask
                mask = (input_ids != 0).bool()  # 创建mask，padding位置为False
                if attention_mask is not None:
                    mask = attention_mask.bool()
                # CRF的损失是负对数似然
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            else:
                # 交叉熵损失
                # 注意：需要将logits和labels reshape为2D
                active_loss = attention_mask.view(-1) == 1 if attention_mask is not None else None
                if active_loss is not None:
                    active_logits = logits.view(-1, self.config["class_num"])[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = self.loss_fn(active_logits, active_labels)
                else:
                    loss = self.loss_fn(logits.view(-1, self.config["class_num"]), labels.view(-1))
            return loss
        else:
            # 预测模式：返回预测结果
            if self.use_crf:
                mask = (input_ids != 0).bool()
                if attention_mask is not None:
                    mask = attention_mask.bool()
                predictions = self.crf.decode(logits, mask=mask)
                return predictions
            else:
                # 直接取argmax
                predictions = torch.argmax(logits, dim=-1)
                return predictions
    
    def predict_with_logits(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        返回logits（用于评估或其他用途）
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = bert_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def choose_optimizer(config, model):
    """选择优化器"""
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    # BERT微调通常使用AdamW
    if optimizer == "adamw":
  
         return AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
   
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")


def create_scheduler(optimizer, config, num_training_steps):
    """创建学习率调度器"""
    warmup_steps = config.get("warmup_steps", 0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler


if __name__ == "__main__":
    from config import Config
    model = BertNERModel(Config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")