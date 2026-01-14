#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import json
from transformers import BertModel, BertTokenizer

"""
基于BERT的SFT语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, bert_model_path="E:\\BaiduNetdiskDownload\\第六周 语言模型\\bert-base-chinese"):
        super(LanguageModel, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        # 获取BERT的隐藏层大小
        self.hidden_size = self.bert.config.hidden_size
        
        # 分类层：将BERT的输出映射到词汇表大小
        self.classify = nn.Linear(self.hidden_size, vocab_size)
        
        # Dropout层
        self.dropout = nn.Dropout(0.1)
        
        # 损失函数
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None, attention_mask=None, sft_mask=None):
        # 通过BERT模型
        outputs = self.bert(x, attention_mask=attention_mask)
        
        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state
        
        # 通过分类层得到预测结果
        y_pred = self.classify(sequence_output)  # shape: (batch_size, seq_len, vocab_size)
        
        if y is not None and sft_mask is not None:
            # 计算损失，只对sft_mask为1的部分计算loss（即s2部分）
            # 将预测值和真实值展平
            y_pred_flat = y_pred.view(-1, y_pred.shape[-1])
            y_flat = y.view(-1)
            sft_mask_flat = sft_mask.view(-1)
            
            # 只对sft_mask为1的位置计算loss
            valid_positions = sft_mask_flat == 1
            if valid_positions.sum() > 0:
                return self.loss(y_pred_flat[valid_positions], y_flat[valid_positions])
            else:
                # 如果没有有效位置，返回0损失
                return torch.tensor(0.0, requires_grad=True, device=y_pred.device)
        elif y is not None:
            # 兼容旧的训练方式
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 返回概率分布
            return torch.softmax(y_pred, dim=-1)

# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 3  # 留出0位给pad token，1位给unk，2位给mask
    return vocab

# 加载语料 - 从sample_data.json
def load_corpus(path):
    corpus = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    corpus.append(data)
                except:
                    continue
    return corpus

# 随机生成一个样本 - SFT格式
def build_sample(vocab, max_length, corpus):
    # 随机选择一个数据项
    data_item = random.choice(corpus)
    title = data_item['title']
    content = data_item['content']
    
    # 构建完整文本：标题 + [SEP] + 内容
    full_text = title + "[SEP]" + content  # 使用[SEP]作为分隔符
    
    # 将文本转换为token ids
    tokens = [vocab.get(char, vocab["<unk>"]) for char in full_text]
    
    # 如果长度超过最大长度，进行截断
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # 构建SFT mask: 标题和分隔符部分为0，内容部分为1
    title_tokens = [vocab.get(char, vocab["<unk>"]) for char in title]
    sep_token = [vocab.get("[SEP]", vocab["<unk>"])] if "[SEP]" in vocab else [vocab["<unk>"]]
    content_tokens = [vocab.get(char, vocab["<unk>"]) for char in content]
    
    # 合并并确保总长度不超过max_length
    combined = title_tokens + sep_token + content_tokens
    if len(combined) > max_length:
        available_content_len = max_length - len(title_tokens) - len(sep_token)
        if available_content_len > 0:
            content_tokens = content_tokens[:available_content_len]
        else:
            combined = title_tokens[:max_length]
            content_tokens = []
    
    # 重新构建x
    x = title_tokens + sep_token + content_tokens
    y = x[:]  # 预测目标是自己
    
    # 构建SFT mask: 标题部分和分隔符为0，内容部分为1
    sft_mask = [0] * len(title_tokens) + [0] * len(sep_token) + [1] * len(content_tokens)
    
    # 如果长度不足max_length，用pad填充
    while len(x) < max_length:
        x.append(vocab["<pad>"])
        y.append(vocab["<pad>"])
        sft_mask.append(0)  # pad部分不参与loss计算
    
    return x, y, sft_mask

# 建立数据集
def build_dataset(sample_length, vocab, max_length, corpus):
    dataset_x = []
    dataset_y = []
    dataset_sft_mask = []
    
    for i in range(sample_length):
        x, y, sft_mask = build_sample(vocab, max_length, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_sft_mask.append(sft_mask)
    
    return (
        torch.LongTensor(dataset_x), 
        torch.LongTensor(dataset_y), 
        torch.LongTensor(dataset_sft_mask)
    )

# 建立模型
def build_model(vocab, bert_model_path="E:\\BaiduNetdiskDownload\\第六周 语言模型\\bert-base-chinese"):
    model = LanguageModel(len(vocab), bert_model_path)
    return model

# 创建SFT式因果掩码
def create_sft_causal_mask(batch_size, seq_len, title_len, device):
    """
    创建SFT式因果掩码：
    - s1 x s1: 全为1（标题内部互相可见）
    - s1 x s2: 全为0（标题对内容不可见）
    - s2 x s1: 全为1（内容对标题可见）
    - s2 x s2: 因果掩码（下三角矩阵，内容内部遵循因果关系）
    """
    mask = torch.zeros((seq_len, seq_len), device=device)
    
    # 确保title_len不超过seq_len
    title_len = min(title_len, seq_len)
    
    # s1部分索引: [0, title_len)
    # [SEP]标记: [title_len, title_len+1) - 如果存在
    # s2部分索引: [title_len+1, seq_len) - 如果存在
    
    if title_len > 0:
        # s1 x s1: 全为1 (标题内部互相可见)
        mask[:title_len, :title_len] = 1
    
    # s1 x s2: 全为0 (标题对内容不可见)
    # 已经初始化为0，无需操作
    
    # [SEP] x all: 全为1 ([SEP]可以看到标题和自己) - 如果[SEP]存在且在范围内
    if title_len < seq_len:
        mask[title_len, :title_len+1] = 1  # [SEP]可以看到自己和标题
    
    # s2 x s1: 全为1 (内容对标题可见) - 如果内容部分存在
    if title_len + 1 < seq_len:
        mask[title_len+1:, :title_len+1] = 1  # 内容可以看到标题和[SEP]
    
    # s2 x s2: 因果掩码（下三角矩阵）- 如果内容部分存在
    s2_start = title_len + 1  # 从标题后一位开始（跳过[SEP]）
    s2_end = seq_len
    s2_len = s2_end - s2_start
    
    if s2_len > 0:
        s2_causal_mask = torch.tril(torch.ones((s2_len, s2_len), device=device))
        mask[s2_start:s2_end, s2_start:s2_end] = s2_causal_mask
    
    # 扩展到batch维度
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    mask = mask.float()
    return mask

# 文本生成测试代码
def generate_sentence(openings, model, vocab, max_length=128):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        generated = openings
        pred_char = ""
        # 生成文本直到达到最大长度或生成换行符
        while len(generated) < max_length:
            # 取输入
            input_text = generated[-max_length:]  # 限制输入长度
            x = [vocab.get(char, vocab["<unk>"]) for char in input_text]
            
            # 确保输入长度为实际长度，不填充到max_length
            actual_len = len(x)
            x_tensor = torch.LongTensor([x])
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
            
            # 创建SFT因果掩码 - 使用实际序列长度和标题长度
            title_len = len(openings)  # 假设开头部分为标题
            # 确保title_len不超过actual_len
            title_len = min(title_len, actual_len)
            causal_mask = create_sft_causal_mask(1, actual_len, title_len, x_tensor.device)
            
            # 获取模型输出
            outputs = model.bert(x_tensor, attention_mask=causal_mask)
            if isinstance(outputs, tuple):
                sequence_output = outputs[0]
            else:
                sequence_output = outputs.last_hidden_state
            y_pred = model.classify(sequence_output)
            
            # 取最后一个位置的预测结果
            y = y_pred[0][-1]
            # 应用softmax确保概率分布非负
            y = torch.softmax(y, dim=-1)
            
            # 更激进的采样策略，增加随机性
            index = sampling_strategy(y, temperature=1.2, top_p=0.85, top_k=50)
            pred_char = reverse_vocab[index]
            
            # 如果生成了换行符或特殊结束符，停止生成
            if pred_char == "\n" or pred_char in ["。", "！", "?"] and len(generated) > len(openings) + 20:
                break  # 生成标点符号且长度足够时停止
                
            generated += pred_char
    
    return generated

def sampling_strategy(prob_distribution, temperature=1.0, top_p=0.9, top_k=0):
    # 应用温度调整
    prob_distribution = torch.pow(prob_distribution, 1.0 / temperature)
    
    # 确保概率分布中的值非负
    prob_distribution = torch.clamp(prob_distribution, min=0.0)
    # 确保概率和为1
    prob_distribution = prob_distribution / prob_distribution.sum()
    
    # Top-k 采样
    if top_k > 0:
        top_k = min(top_k, prob_distribution.size(-1))
        indices_to_remove = prob_distribution < torch.topk(prob_distribution, top_k)[0][..., -1, None]
        prob_distribution[indices_to_remove] = 0
        prob_distribution = prob_distribution / prob_distribution.sum()
    
    # Top-p (nucleus) 采样
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(prob_distribution, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 只保留累积概率小于top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        prob_distribution[indices_to_remove] = 0
        prob_distribution = prob_distribution / prob_distribution.sum()
    
    # 随机采样
    prob_distribution = prob_distribution.cpu().numpy()
    return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

# 计算文本ppl
def calc_perplexity(sentence, model, vocab, max_length=128):
    prob = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - max_length)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<unk>"]) for char in window]
            
            # 使用实际长度
            actual_len = len(x)
            x_tensor = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<unk>"])
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
            
            # 创建SFT因果掩码
            title_len = min(len(sentence[:start]), actual_len)  # 使用实际输入长度
            causal_mask = create_sft_causal_mask(1, actual_len, title_len, x_tensor.device)
            
            # 获取模型输出
            outputs = model.bert(x_tensor, attention_mask=causal_mask)
            if isinstance(outputs, tuple):
                sequence_output = outputs[0]
            else:
                sequence_output = outputs.last_hidden_state
            y_pred = model.classify(sequence_output)
            
            # 取最后一个位置的预测概率
            pred_prob_distribute = torch.softmax(y_pred[0][-1], dim=-1)  # 应用softmax
            target_prob = pred_prob_distribute[target_index]
            if target_prob > 0:
                prob += math.log(target_prob.item())
                count += 1
    
    if count > 0:
        return math.exp(-prob / count)
    else:
        return float('inf')

def train(corpus_path, save_weight=True):
    epoch_num = 30  # 增加训练轮数
    batch_size = 2  # 进一步减小batch size以减少过拟合
    train_sample = 500  # 减少每轮训练样本数
    max_length = 128  # 样本文本最大长度
    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 使用更小的学习率和更强的正则化
    optim = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.05)  # 更小的学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=8, gamma=0.9)
    
    print("BERT SFT语言模型加载完毕，开始训练")
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, sft_mask = build_dataset(batch_size, vocab, max_length, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y, sft_mask = x.cuda(), y.cuda(), sft_mask.cuda()
            
            # 创建SFT因果掩码 - 需要动态确定标题长度
            batch_masks = []
            for i in range(batch_size):
                # 找到[SEP]标记的位置，如果没有，则使用标题的固定长度
                sep_token_id = vocab.get("[SEP]", -1)
                if sep_token_id != -1:
                    sep_pos = (x[i] == sep_token_id).nonzero(as_tuple=True)[0]
                    if len(sep_pos) > 0:
                        title_len = sep_pos[0].item()
                    else:
                        title_len = min(10, x.size(1)//3)  # 假设标题长度不超过总长度的1/3
                else:
                    title_len = min(10, x.size(1)//3)  # 默认标题长度
                    
                mask = create_sft_causal_mask(1, x.size(1), title_len, x.device)[0]  # 取单个mask
                batch_masks.append(mask)
            
            causal_mask = torch.stack(batch_masks, dim=0)
            
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, attention_mask=causal_mask, sft_mask=sft_mask)  # 计算loss，使用SFT mask
            loss.backward()  # 计算梯度
            
            # 更强的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        
        scheduler.step()  # 学习率调度
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        
        # 每5轮测试一次生成
        if (epoch + 1) % 5 == 0:
            # 测试生成 - 使用标题作为开头
            test_titles = [
                "阿根廷歹徒抢服装尺码不对拿回店里换",
                "国际通用航空大会沈阳飞行家表演队一飞机发生坠机，伤亡不明",
                "6月1日起北京实施“史上最严控烟期”"
            ]
            for title in test_titles:
                generated = generate_sentence(title, model, vocab, max_length)
                print(f"输入标题: {title}")
                print(f"生成内容: {generated[len(title):]}")
                print("-" * 50)
    
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
