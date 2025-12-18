modle.py------------------------------------------------------------------------------------------------------------------------------------modle.py
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



loader.py--------------------------------------------------------------------------------------------------------------------

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


train.py---------------------------------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
import time
from datetime import datetime
from config import Config
from model import BertNERModel, choose_optimizer, create_scheduler
from evaluate import Evaluator
from loader import load_data, get_tokenizer

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 配置日志
def setup_logger():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"train_{current_time}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scheduler, device, config, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    step = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if config.get("max_grad_norm", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        
        # 优化器步进
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 记录损失
        total_loss += loss.item()
        step += 1
        
        # 打印进度
        if batch_idx % 50 == 0:
            avg_loss = total_loss / step
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_model(model, epoch, config, is_best=False):
    """保存模型"""
    model_dir = config["model_path"]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if is_best:
        model_path = os.path.join(model_dir, "best_model.pth")
    else:
        model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
    
    # 保存模型状态字典和配置
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    
    logger.info(f"模型保存到: {model_path}")


def main(config):
    """主训练函数"""
    global logger
    logger = setup_logger()
    
    # 设置随机种子
    set_seed(config.get("seed", 42))
    
    logger.info("开始训练...")
    logger.info(f"配置参数: {config}")
    
    # 创建输出目录
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])
    
    # 获取设备
    device = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer = get_tokenizer(config)
    
    # 加载数据
    logger.info("加载训练数据...")
    train_dataloader, train_dataset = load_data(
        config["train_data_path"], 
        config, 
        tokenizer, 
        shuffle=True
    )
    
    logger.info("加载验证数据...")
    valid_dataloader, valid_dataset = load_data(
        config["valid_data_path"], 
        config, 
        tokenizer, 
        shuffle=False
    )
    
    # 初始化模型
    logger.info("初始化模型...")
    model = BertNERModel(config)
    model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    
    # 初始化优化器
    optimizer = choose_optimizer(config, model)
    
    # 计算总的训练步数
    total_steps = len(train_dataloader) * config["epoch"]
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # 初始化评估器
    evaluator = Evaluator(config, model, tokenizer, logger, device)
    
    # 训练循环
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, config["epoch"] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{config['epoch']}")
        logger.info(f"{'='*50}")
        
        # 训练一个epoch
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, config, epoch)
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch} 训练完成, 平均损失: {train_loss:.4f}, 耗时: {epoch_time:.2f}秒")
        
        # 评估模型
        logger.info(f"开始评估 Epoch {epoch}...")
        eval_results = evaluator.evaluate(valid_dataloader, epoch)
        
        # 获取F1分数
        current_f1 = eval_results.get('micro_f1', 0)
        
        # 保存最佳模型
        if config.get("save_best_model", True) and current_f1 > best_f1:
            best_f1 = current_f1
            save_model(model, epoch, config, is_best=True)
            patience_counter = 0
            logger.info(f"新的最佳模型，F1分数: {best_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"F1分数未提升，当前最佳: {best_f1:.4f}")
        
        # 定期保存模型
        if epoch % 5 == 0:
            save_model(model, epoch, config, is_best=False)
        
        # 早停检查
        if patience_counter >= config.get("early_stop_patience", 5):
            logger.info(f"早停触发，连续 {patience_counter} 个epoch未提升")
            break
    
    logger.info(f"\n训练完成，最佳F1分数: {best_f1:.4f}")
    
    # 加载最佳模型
    best_model_path = os.path.join(config["model_path"], "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"加载最佳模型: {best_model_path}")
    
    return model, tokenizer


if __name__ == "__main__":
    # 开始训练
    start_time = time.time()
    model, tokenizer = main(Config)
    total_time = time.time() - start_time
    logger.info(f"总训练时间: {total_time:.2f}秒")












