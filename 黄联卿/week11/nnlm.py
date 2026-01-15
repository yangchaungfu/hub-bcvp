#coding:utf8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertTokenizer, BertModel
"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrained_model_path, max_title_len=32):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path, return_dict=False)
        self.max_title_len = max_title_len
        # 添加一个特殊的开始解码token的embedding
        self.decoder_start_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding

    def create_block_diagonal_mask(self, batch_size, seq_len, title_lens):
        mask = torch.zeros((batch_size, seq_len, seq_len))

        for b in range(batch_size):
            title_len = min(title_lens[b].item(), self.max_title_len)

            # 标题部分：双向注意力
            mask[b, :title_len, :title_len] = 1

            # 正文部分：因果注意力（可看到标题+前面的正文）
            mask[b, title_len:, :title_len] = 1  # 正文可看到标题
            causal_mask = torch.tril(torch.ones(seq_len - title_len, seq_len - title_len))
            mask[b, title_len:, title_len:] = causal_mask

        return mask

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, title_lens=None):
        if y is not None:
            # 训练时，使用分块对角线mask
            if title_lens is None:
                # 如果没有提供标题长度，默认使用最大标题长度
                title_lens = torch.full((x.shape[0],), self.max_title_len, dtype=torch.long)

            mask = self.create_block_diagonal_mask(x.shape[0], x.shape[1], title_lens)

            # print(mask, mask.shape)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载语料
def load_json_data(file_path):
    """加载JSON格式的数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出

def build_sample(tokenizer, max_seq_len, data_item):
    """构建训练样本：标题+正文"""
    title = data_item['title']
    content = data_item['content']

    # 编码标题和正文
    title_tokens = tokenizer.encode(title, add_special_tokens=False)
    content_tokens = tokenizer.encode(content, add_special_tokens=False)

    # 限制标题长度
    title_tokens = title_tokens[:32]

    # 合并标题和正文
    combined_tokens = title_tokens + content_tokens

    # 限制总长度
    if len(combined_tokens) > max_seq_len:
        combined_tokens = combined_tokens[:max_seq_len]

    # 如果不够长度，填充padding
    if len(combined_tokens) < max_seq_len:
        combined_tokens.extend([0] * (max_seq_len - len(combined_tokens)))
    else:
        combined_tokens = combined_tokens[:max_seq_len]

    # 创建目标序列（错位一位）
    target_tokens = combined_tokens[1:] + [0]  # 最后一位用padding填充

    return combined_tokens, target_tokens, len(title_tokens)

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, max_seq_len, data_list):
    dataset_x = []
    dataset_y = []
    title_lengths = []

    for i in range(min(sample_length, len(data_list))):
        x, y, title_len = build_sample(tokenizer, max_seq_len, data_list[i])
        dataset_x.append(x)
        dataset_y.append(y)
        title_lengths.append(title_len)

    return (torch.LongTensor(dataset_x),
            torch.LongTensor(dataset_y),
            torch.LongTensor(title_lengths))

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence_with_title(title, model, tokenizer, max_length=200):
    """根据标题生成正文"""
    model.eval()
    with torch.no_grad():
        # 编码标题
        title_tokens = tokenizer.encode(title, add_special_tokens=False)
        current_input = title_tokens.copy()

        generated_content = ""

        for _ in range(max_length):
            if len(current_input) >= 512:  # BERT的最大长度限制
                break

            x = torch.LongTensor([current_input])
            if torch.cuda.is_available():
                x = x.cuda()

            y_pred = model(x)  # 获取概率分布
            next_token_prob = y_pred[0][-1]  # 取最后一个位置的预测

            # 采样下一个token
            next_token_id = sampling_strategy(next_token_prob)

            if next_token_id == tokenizer.pad_token_id or next_token_id == tokenizer.sep_token_id:
                break

            current_input.append(next_token_id)
            next_char = tokenizer.decode([next_token_id])

            # 过滤特殊字符
            if next_char not in ['[PAD]', '[CLS]', '[SEP]']:
                generated_content += next_char

            # 如果生成了换行符或句号，可以考虑停止
            if next_char in ['\n', '。'] and len(generated_content) > len(title):
                break

    return title + "\n" + generated_content

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train_sft(train_path, valid_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数（降低以适应更长序列）
    train_sample = 2000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    max_seq_len = 256  # 最大序列长度（标题+正文）
    vocab_size = 21128  # 字表大小
    learning_rate = 0.0001  # 学习率降低以适应更复杂的任务

    pretrain_model_path = r'E:\newlife\badou\第六周 语言模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # 加载训练和验证数据
    train_data = load_json_data(train_path)
    valid_data = load_json_data(valid_path)

    print(f"训练数据量: {len(train_data)}, 验证数据量: {len(valid_data)}")

    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("SFT模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 训练阶段
        for batch_idx in range(int(train_sample / batch_size)):
            start_idx = (batch_idx * batch_size) % len(train_data)
            end_idx = min(start_idx + batch_size, len(train_data))
            current_batch_data = train_data[start_idx:end_idx]

            if len(current_batch_data) == 0:
                continue

            x, y, title_lens = build_dataset(len(current_batch_data), tokenizer, max_seq_len, current_batch_data)

            if torch.cuda.is_available():
                x, y, title_lens = x.cuda(), y.cuda(), title_lens.cuda()

            optim.zero_grad()  # 梯度归零
            loss = model(x, y, title_lens)  # 计算loss，传入标题长度
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss) if watch_loss else float('inf')
        print(f"=========\n第{epoch + 1}轮平均loss: {avg_loss}")

        # 测试生成效果
        if valid_data:
            test_titles = [
                "娱乐",
                "财经",
                "科技"
            ]

            for title in test_titles[:2]:  # 只测试前两个标题
                generated = generate_sentence_with_title(
                    f"{title}",
                    model,
                    tokenizer
                )
                print(f"\n标题: {title}")
                print(f"生成内容: {generated[len(title):][:200]}...")  # 只显示部分内容
    if not save_weight:
        return
    else:
        base_name = os.path.basename(train_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train_path = r".\data\train_tag_news.json"
    valid_path = r".\data\valid_tag_news.json"

    train_sft(train_path, valid_path, True)
