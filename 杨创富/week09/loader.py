# -*- coding: utf-8 -*-

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer

from transformers import AutoTokenizer, BertTokenizerFast

class NERDataset(Dataset):
    """命名实体识别数据集"""
    
    def __init__(self, data_path, config, tokenizer, is_train=True):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config["max_length"]
        self.is_train = is_train
        
        # 加载标签映射
        self.label2id = self.load_schema(config["schema_path"])
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 加载数据
        self.data = self.load_data(data_path)
        
    def load_schema(self, schema_path):
        """加载标签映射"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    
    def load_data(self, data_path):
        """加载数据文件"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            sentences = f.read().strip().split('\n\n')
            
        for sentence in sentences:
            lines = sentence.strip().split('\n')
            if not lines:
                continue
                
            tokens = []
            labels = []
            
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    tokens.append(token)
                    labels.append(label)
            
            if tokens and labels:
                data.append({
                    'tokens': tokens,
                    'labels': labels
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']
        
        # 对标签进行编码
        label_ids = [self.label2id.get(label, -1) for label in labels]
        
        # 使用BERT tokenizer对文本进行编码
        # 注意：需要特殊处理subword对齐问题
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 创建word_ids映射（解决subword对齐问题）
        word_ids = encoded.word_ids()
        
        # 调整标签以匹配tokenized结果
        aligned_label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊token：[CLS], [SEP], [PAD]
                aligned_label_ids.append(-1)
            elif word_idx != previous_word_idx:
                # 当前token是某个词的第一个subword
                aligned_label_ids.append(label_ids[word_idx])
            else:
                # 当前token是某个词的非第一个subword
                # 对于BIO标注，如果是B-标签，后续subword应该变成I-标签
                if aligned_label_ids[-1] >= 0 and self.id2label[aligned_label_ids[-1]].startswith('B-'):
                    # 将B-标签改为I-标签
                    label_name = self.id2label[aligned_label_ids[-1]]
                    i_label_name = 'I-' + label_name[2:]
                    aligned_label_ids.append(self.label2id.get(i_label_name, aligned_label_ids[-1]))
                else:
                    aligned_label_ids.append(aligned_label_ids[-1])
            previous_word_idx = word_idx
        
        # 转换为tensor
        label_tensor = torch.tensor(aligned_label_ids, dtype=torch.long)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0),
            'labels': label_tensor,
            'original_tokens': tokens,
            'original_labels': labels
        }


def collate_fn(batch):
    """自定义批处理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
    }


def load_data(data_path, config, tokenizer, shuffle=True):
    """加载数据并创建DataLoader"""
    dataset = NERDataset(data_path, config, tokenizer, is_train=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2
    )
    return dataloader, dataset


from transformers import AutoTokenizer, BertTokenizerFast

def get_tokenizer(config):
    """获取BERT tokenizer"""
    try:
        # 方法1：尝试使用BertTokenizerFast
        print("尝试加载BertTokenizerFast...")
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        print(f"✓ 加载成功: {type(tokenizer).__name__}")
        return tokenizer
    except Exception as e:
        print(f"BertTokenizerFast失败: {e}")
        
        # 方法2：尝试AutoTokenizer
        try:
            print("尝试AutoTokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                config["bert_path"], 
                use_fast=True,
                tokenizer_type="bert"
            )
            print(f"✓ 加载成功: {type(tokenizer).__name__}")
            return tokenizer
        except Exception as e2:
            print(f"AutoTokenizer也失败: {e2}")
            
            # 方法3：使用慢速tokenizer但修改数据处理
            print("使用慢速tokenizer，将修改数据处理方式...")
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
            return tokenizer


if __name__ == "__main__":
    from config import Config
    tokenizer = get_tokenizer(Config)
    dataloader, dataset = load_data(Config["train_data_path"], Config, tokenizer)
    
    print(f"数据集大小: {len(dataset)}")
    batch = next(iter(dataloader))
    print(f"批次数据形状:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")