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