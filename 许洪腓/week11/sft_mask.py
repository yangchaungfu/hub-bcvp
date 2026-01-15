import torch
import torch.nn as nn
import numpy as np
import math
import random
import os 
import re 
from transformers import BertModel, BertTokenizer
import json 

"""
1. 修改load_corpus
2. 修改data的内容，使输入给模型的数据满足SFT的要求：输入和输出错位，但要掩码
3. 最后增加输入的掩码
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab, tokenizer):
        super().__init__()
        self.bert = BertModel.from_pretrained(r"E:\models\bert-base-chinese")
        hidden_size = self.bert.config.hidden_size
        self.tokenizer = tokenizer
        self.classify = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def _build_qa_causal_mask(self, x, sep_token_id):
        batch_size, seq_len = x.shape 
        padding_mask = torch.where(x != 0, torch.tensor(1), torch.tensor(0))
        padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)

        sep_pos_list = []
        for i in range(batch_size):
            sep_pos = (x[i]== sep_token_id).nonzero(as_tuple=True)[0]
            sep_pos = sep_pos[0].item() 
            sep_pos_list.append(sep_pos)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        qa_mask = torch.ones_like(causal_mask)
        for i in range(batch_size):
            sep_pos = sep_pos_list[i]
            qa_mask[i, 0:sep_pos+1, sep_pos+1:] = 0
            qa_mask[i, sep_pos+1:, sep_pos+1:] = causal_mask[i, sep_pos+1:, sep_pos+1:]

        final_mask = qa_mask * padding_mask
        return final_mask

    def forward(self, x, y=None):
        sep_token_id = self.tokenizer.sep_token_id
        attention_mask = self._build_qa_causal_mask(x, sep_token_id)
        output = self.bert(x, attention_mask=attention_mask)
        y_pred = self.classify(output.last_hidden_state)
        if y is not None:
            loss_mask = torch.where(y != 0, torch.tensor(1), torch.tensor(0))
            loss = self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), reduction="none")
            loss = (loss * loss_mask.view(-1)).sum()
            loss = loss / loss_mask.sum()
            return loss 
        else:
            return torch.softmax(y_pred, dim=-1)
        
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            char = line[:-1]
            vocab[char] = index + 1
    return vocab 

def load_qa_pairs(path):
    qa_pairs = []
    with open(path, encoding="utf8") as f:
        for line in f:
            text = json.loads(line)
            title = text["title"]
            content = text["content"]
            qa_pairs.append((title, content))
    return qa_pairs

def build_sample(tokenizer, max_sep_len, qa_pairs):
    q, a = random.choice(qa_pairs)
    input_text = f"{q}{tokenizer.sep_token}{a}"

    encoded = tokenizer(
        input_text,
        truncation=True,
        max_length=max_sep_len,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].squeeze(0)   

    target_ids = torch.cat([input_ids[1:], torch.tensor([0], dtype=torch.long)],dim=0)

    sep_token_id = tokenizer.sep_token_id
    sep_pos = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    sep_pos = sep_pos[0].item()
    target_ids[:sep_pos+1] = 0 
    return input_ids, target_ids       

    

def build_dataset(sample_length, tokenizer, max_sep_len, qa_pairs):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, max_sep_len, qa_pairs)
        dataset_x.append(x.numpy().tolist())
        dataset_y.append(y.numpy().tolist())
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, tokenizer):
    model = LanguageModel(vocab, tokenizer)
    return model 

def generate_sentence(prompt, model, tokenizer, max_gen_len=300):
    """重构生成函数：统一Tokenizer编码，适配SFT模型，支持问题→答案生成"""
    model.eval()
    generated_text = prompt
    sep_token = tokenizer.sep_token
    # 确保prompt末尾带[SEP]（分隔问题和答案）
    if not generated_text.endswith(sep_token):
        generated_text += sep_token

    with torch.no_grad():
        while len(generated_text) < max_gen_len:
            # 用Tokenizer统一编码（和训练阶段一致，避免[SEP]编码错误）
            encoded = tokenizer(
                generated_text,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"]
            # 模型预测
            pred_probs = model(input_ids)[0][-1]  # 取最后一个token的预测概率
            next_token_id = sampling_strategy(pred_probs)
            next_token = tokenizer.decode([next_token_id])  # 解码为文本
            
            # 终止条件：生成PAD/SEP/换行符则停止
            if next_token in [tokenizer.pad_token, tokenizer.sep_token, "\n"]:
                break
            generated_text += next_token
    # 过滤生成结果中的特殊符号，返回纯答案
    return generated_text

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))))
    

def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0 
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i-window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))

def train(corpus_path, save_weight=True):
    epoch_num = 20 
    batch_size = 64 
    train_sample = 5000
    vocab = build_vocab(r"E:\models\bert-base-chinese\vocab.txt")
    tokenizer = BertTokenizer.from_pretrained(r"E:\models\bert-base-chinese")
    qa_pairs = load_qa_pairs(corpus_path)
    model = build_model(vocab, tokenizer)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, 100 , qa_pairs)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer))
        print(generate_sentence("国际通用航空大会沈阳飞行家表演队一飞机发生坠机，伤亡不明", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    train("sample_data.json", False)
