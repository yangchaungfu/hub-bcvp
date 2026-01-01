#coding:utf8

import torch
import torch.nn as nn
import numpy as np
from config import Config
from loader import load_vocab, load_corpus
from torch.optim import Adam, SGD
from transformers import BertModel, BertTokenizer

"""
基于BERT的Seq2Seq模型，用于监督微调(SFT)训练
实现从title到content的生成任务
"""

def subsequent_mask(size):
    """
    生成因果掩码(Causal Mask)，用于防止解码器在预测时看到未来的token
    
    参数:
        size: 序列长度
    返回:
        形状为(1, size, size)的布尔张量，上三角部分为False（被遮蔽）
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def pad_mask(inputs, pad_id=0):
    """
    生成填充掩码(Padding Mask)，用于忽略序列中的padding token
    
    参数:
        inputs: 输入的张量，形状为(batch_size, seq_len)
        pad_id: padding token的索引，默认为0
    返回:
        形状为(batch_size, 1, seq_len)的布尔张量，padding位置为False
    """
    return (inputs != pad_id).unsqueeze(1)

class Seq2SeqModel(nn.Module):
    def __init__(self, config: Config):
        super(Seq2SeqModel, self).__init__()
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2  # 序列开始标记
        self.eos_id = 3  # 序列结束标记
        
        # 保存配置
        self.config = config
        
        # 加载BERT模型作为编码器和解码器的共享编码器
        self.bert = BertModel.from_pretrained(config["bert_path"])
        
        # 加载词表和语料
        self.vocab = load_vocab(config["vocab_path"])
        self.data_list = load_corpus(config["corpus_path"])
        
        # 输出层：将BERT的隐藏状态映射到词表大小
        self.classifier = nn.Linear(config["hidden_size"], len(self.vocab))
        
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, input_seq, target_seq=None):
        """
        前向传播函数，实现Seq2Seq的编码-解码过程
        
        参数:
            input_seq: 输入序列(title)，形状为(batch_size, input_seq_len)
            target_seq: 目标序列(content)，用于计算loss，形状为(batch_size, target_seq_len)
        返回:
            如果target_seq存在，返回loss值；否则返回预测的概率分布
        """
        # 生成输入序列的padding掩码
        input_mask = pad_mask(input_seq, self.pad_id)
        
        if target_seq is not None:
            # 训练阶段：使用teacher forcing
            
            # 生成目标序列的padding掩码
            target_mask = pad_mask(target_seq, self.pad_id)
            
            # 生成因果掩码，防止解码器看到未来的token
            causal_mask = subsequent_mask(target_seq.size(1)).to(target_seq.device)
            
            # 合并掩码
            combined_mask = target_mask & causal_mask
            
            # 使用BERT处理输入序列（编码）
            encoder_output, _ = self.bert(input_seq, attention_mask=input_mask)
            
            # 使用BERT处理目标序列（解码）
            decoder_output, _ = self.bert(target_seq, attention_mask=combined_mask)
            
            # 计算预测概率
            logits = self.classifier(decoder_output)
            
            # 计算损失，忽略padding位置
            loss = self.loss(logits.view(-1, logits.shape[-1]), target_seq.view(-1), 
                           ignore_index=self.pad_id)
            
            return loss
        else:
            # 推理阶段：自回归生成
            batch_size = input_seq.size(0)
            device = input_seq.device
            
            # 使用BERT处理输入序列（编码）
            encoder_output, _ = self.bert(input_seq, attention_mask=pad_mask(input_seq, self.pad_id))
            
            # 初始化输出序列，以SOS标记开始
            output_seq = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=device)
            
            # 用于存储所有预测的logits
            all_logits = []
            
            # 自回归生成
            for _ in range(self.config["max_output_len"]):  # 使用配置中的最大输出长度
                # 生成当前输出序列的掩码
                output_mask = pad_mask(output_seq, self.pad_id)
                causal_mask = subsequent_mask(output_seq.size(1)).to(device)
                combined_mask = output_mask & causal_mask
                
                # 使用BERT处理当前输出序列
                decoder_output, _ = self.bert(output_seq, attention_mask=combined_mask)
                
                # 获取最后一个token的logits
                last_logits = self.classifier(decoder_output[:, -1, :])
                all_logits.append(last_logits.unsqueeze(1))
                
                # 预测下一个token
                next_token = torch.argmax(last_logits, dim=-1).unsqueeze(1)
                
                # 将预测的token添加到输出序列
                output_seq = torch.cat([output_seq, next_token], dim=1)
                
                # 如果所有序列都生成了EOS标记，停止生成
                if (next_token == self.eos_id).all():
                    break
            
            # 合并所有logits
            logits = torch.cat(all_logits, dim=1)
            
            return torch.softmax(logits, dim=-1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)