# -*- coding: utf-8 -*-

import torch
import numpy as np
import re
from collections import defaultdict
from sklearn.metrics import classification_report, precision_recall_fscore_support

class Evaluator:
    def __init__(self, config, model, tokenizer, logger, device):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.device = device
        
        # 加载标签映射
        self.label2id = self.load_schema(config["schema_path"])
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.class_num = config["class_num"]
        
        # 实体类型列表
        self.entity_types = ['PER', 'LOC', 'ORG', 'TIME']
        
    def load_schema(self, schema_path):
        """加载标签映射"""
        import json
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    

    
    def align_predictions(self, predictions, label_ids, attention_mask):
        """对齐预测结果，处理subword问题"""
        true_labels = []
        pred_labels = []
        
        for i in range(len(predictions)):
            seq_len = attention_mask[i].sum().item()
            
            # 获取有效部分的真实标签
            seq_true = label_ids[i][:seq_len].cpu().numpy()
            
            # 获取预测标签
            if isinstance(predictions[i], list):
                # 如果是列表，确保长度与seq_len一致
                seq_pred = predictions[i][:seq_len] if len(predictions[i]) >= seq_len else predictions[i]
            else:
                # 如果是张量或数组
                seq_pred = predictions[i][:seq_len]
                if torch.is_tensor(seq_pred):
                    seq_pred = seq_pred.cpu().numpy()
            
            # 过滤掉特殊token（[CLS], [SEP], [PAD]）
            mask = seq_true != -1
            
            # 确保seq_pred的长度与mask一致
            if isinstance(seq_pred, list):
                # 列表的情况
                if len(seq_pred) == len(mask):
                    seq_pred_filtered = [seq_pred[j] for j in range(len(seq_pred)) if mask[j]]
                    pred_labels.extend(seq_pred_filtered)
                else:
                    # 长度不匹配，跳过或处理
                    valid_indices = [j for j in range(min(len(seq_pred), len(mask))) if mask[j]]
                    seq_pred_filtered = [seq_pred[j] for j in valid_indices]
                    pred_labels.extend(seq_pred_filtered)
            else:
                # numpy数组的情况
                if len(seq_pred) == len(mask):
                    pred_labels.extend(seq_pred[mask])
                else:
                    # 长度不匹配，取交集
                    min_len = min(len(seq_pred), len(mask))
                    pred_labels.extend(seq_pred[:min_len][mask[:min_len]])
            
            true_labels.extend(seq_true[mask])
        
        return true_labels, pred_labels


 
    def evaluate(self, dataloader, epoch=None):
        """评估模型"""
        if epoch:
            self.logger.info(f"开始评估第{epoch}轮模型...")
        
        self.model.eval()
        
        # 统计结果
        all_true_labels = []
        all_pred_labels = []
        entity_stats = defaultdict(lambda: defaultdict(int))
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 获取预测
                if self.config["use_crf"]:
                    predictions = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    # 确保predictions是列表的列表
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.cpu().numpy().tolist()
                else:
                    logits = self.model.predict_with_logits(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                
                # 对齐标签
                batch_true, batch_pred = self.align_predictions(
                    predictions,
                    labels.cpu(),
                    attention_mask.cpu()
                )
                
                all_true_labels.extend(batch_true)
                all_pred_labels.extend(batch_pred)
                
                # 实体级别的评估（只处理前几个batch）
                if batch_idx < 3:
                    self.calculate_entity_metrics(batch, predictions, entity_stats)


        # 计算分类报告
        self.logger.info("\n" + "="*50)
        self.logger.info("分类报告 (Token级别):")
        self.logger.info("="*50)
        
        # 过滤掉O标签（索引为0）
        mask = np.array(all_true_labels) != 0
        filtered_true = np.array(all_true_labels)[mask]
        filtered_pred = np.array(all_pred_labels)[mask]
        
        if len(filtered_true) > 0:
            # 计算每个类别的指标
            target_names = [self.id2label.get(i, f"LABEL_{i}") for i in range(self.class_num)]
            
            report = classification_report(
                filtered_true,
                filtered_pred,
                labels=list(range(self.class_num)),
                target_names=target_names,
                zero_division=0
            )
            self.logger.info(report)
        
        # 计算总体指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='weighted', zero_division=0
        )
        
        # 计算micro F1
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='micro', zero_division=0
        )
        
        # 计算macro F1
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='macro', zero_division=0
        )
        
        self.logger.info("\n总体指标:")
        self.logger.info(f"加权平均 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        self.logger.info(f"Micro平均 - 精确率: {micro_precision:.4f}, 召回率: {micro_recall:.4f}, F1分数: {micro_f1:.4f}")
        self.logger.info(f"Macro平均 - 精确率: {macro_precision:.4f}, 召回率: {macro_recall:.4f}, F1分数: {macro_f1:.4f}")
        
        # 打印实体级别的统计
        self.logger.info("\n实体级别统计:")
        for entity_type in self.entity_types:
            stats = entity_stats[entity_type]
            if stats['true'] > 0 or stats['pred'] > 0:
                precision = stats['correct'] / max(stats['pred'], 1)
                recall = stats['correct'] / max(stats['true'], 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                self.logger.info(f"{entity_type}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
                self.logger.info(f"  正确识别: {stats['correct']}, 样本实体数: {stats['true']}, 识别出实体数: {stats['pred']}")
        
        return {
            'weighted_f1': f1,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'precision': precision,
            'recall': recall
        }
    
    def calculate_entity_metrics(self, batch, predictions, entity_stats):
        """计算实体级别的指标"""
        # 获取原始数据
        original_tokens = batch.get('original_tokens', None)
        original_labels = batch.get('original_labels', None)
        
        if original_tokens is None or original_labels is None:
            return
        
        for i in range(len(original_tokens)):
            tokens = original_tokens[i]
            true_labels = original_labels[i]
            
            # 提取真实实体
            true_entities = self.extract_entities_from_labels(tokens, true_labels)
            
            # 提取预测实体（需要将预测结果映射回原始token）
            # 注意：这里简化处理，实际需要处理subword对齐
            pred_labels = predictions[i][:len(tokens)]
            pred_entities = self.extract_entities_from_labels(tokens, pred_labels)
            
            # 统计
            for entity_type in self.entity_types:
                entity_stats[entity_type]['true'] += len(true_entities.get(entity_type, []))
                entity_stats[entity_type]['pred'] += len(pred_entities.get(entity_type, []))
                entity_stats[entity_type]['correct'] += len([
                    ent for ent in pred_entities.get(entity_type, []) 
                    if ent in true_entities.get(entity_type, [])
                ])
    
    def extract_entities_from_labels(self, tokens, labels):
        """从标签序列中提取实体"""
        entities = defaultdict(list)
        current_entity = []
        current_type = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if isinstance(label, str):
                label_str = label
            else:
                label_str = self.id2label.get(label, 'O')
            
            if label_str.startswith('B-'):
                # 保存前一个实体
                if current_entity:
                    entities[current_type].append(''.join(current_entity))
                
                # 开始新实体
                current_entity = [token]
                current_type = label_str[2:]  # 去掉"B-"
            
            elif label_str.startswith('I-'):
                # 继续当前实体
                if current_type and label_str[2:] == current_type:
                    current_entity.append(token)
                else:
                    # 非法序列，结束当前实体
                    if current_entity:
                        entities[current_type].append(''.join(current_entity))
                    current_entity = []
                    current_type = None
            
            else:  # 'O'
                # 结束当前实体
                if current_entity:
                    entities[current_type].append(''.join(current_entity))
                current_entity = []
                current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entities[current_type].append(''.join(current_entity))
        
        return entities


if __name__ == "__main__":
    # 测试评估器
    from config import Config
    from model import BertNERModel
    from loader import get_tokenizer, load_data
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(Config)
    
    model = BertNERModel(Config)
    model.to(device)
    
    dataloader, _ = load_data(Config["valid_data_path"], Config, tokenizer, shuffle=False)
    
    evaluator = Evaluator(Config, model, tokenizer, logger, device)
    results = evaluator.evaluate(dataloader)
    
    print(f"评估完成，Micro-F1: {results['micro_f1']:.4f}")