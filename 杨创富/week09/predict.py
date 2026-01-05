# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
from model import BertNERModel
from loader import get_tokenizer
from config import Config

class NERPredictor:
    """命名实体识别预测器"""
    
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
        
        # 加载tokenizer
        self.tokenizer = get_tokenizer(config)
        
        # 加载标签映射
        self.label2id = self.load_schema(config["schema_path"])
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 加载模型
        self.model = BertNERModel(config)
        
        if model_path:
            self.load_model(model_path)
        else:
            # 加载最佳模型
            best_model_path = f"{config['model_path']}/best_model.pth"
            if os.path.exists(best_model_path):
                self.load_model(best_model_path)
            else:
                raise FileNotFoundError(f"未找到模型文件: {best_model_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_schema(self, schema_path):
        """加载标签映射"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    
    def load_model(self, model_path):
        """加载模型权重"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功: {model_path}")
    
    def predict(self, text):
        """预测单个文本"""
        # 分词
        tokens = list(text)  # 按字分割
        
        # 编码
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors='pt'
        )
        
        # 移动到设备
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_type_ids = encoded['token_type_ids'].to(self.device)
        
        # 预测
        with torch.no_grad():
            if self.config["use_crf"]:
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )[0]  # 取第一个样本
            else:
                logits = self.model.predict_with_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()
        
        # 获取有效预测
        word_ids = encoded.word_ids()
        aligned_predictions = []
        
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                # 特殊token
                continue
            elif word_idx != previous_word_idx:
                # 当前token是某个词的第一个subword
                aligned_predictions.append(predictions[i])
            previous_word_idx = word_idx
        
        # 确保长度与输入一致
        aligned_predictions = aligned_predictions[:len(tokens)]
        
        # 提取实体
        entities = self.extract_entities(tokens, aligned_predictions)
        
        return {
            'text': text,
            'tokens': tokens,
            'predictions': [self.id2label.get(p, 'O') for p in aligned_predictions],
            'entities': entities
        }
    
    def extract_entities(self, tokens, predictions):
        """从预测结果中提取实体"""
        entities = []
        current_entity = []
        current_type = None
        start_pos = 0
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            pred_label = self.id2label.get(pred_id, 'O')
            
            if pred_label.startswith('B-'):
                # 保存前一个实体
                if current_entity:
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type,
                        'start': start_pos,
                        'end': start_pos + len(''.join(current_entity))
                    })
                
                # 开始新实体
                current_entity = [token]
                current_type = pred_label[2:]
                start_pos = i
            
            elif pred_label.startswith('I-'):
                # 继续当前实体
                if current_type and pred_label[2:] == current_type:
                    current_entity.append(token)
                else:
                    # 非法序列，结束当前实体
                    if current_entity:
                        entities.append({
                            'text': ''.join(current_entity),
                            'type': current_type,
                            'start': start_pos,
                            'end': start_pos + len(''.join(current_entity))
                        })
                    current_entity = []
                    current_type = None
            
            else:  # 'O'
                # 结束当前实体
                if current_entity:
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type,
                        'start': start_pos,
                        'end': start_pos + len(''.join(current_entity))
                    })
                current_entity = []
                current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append({
                'text': ''.join(current_entity),
                'type': current_type,
                'start': start_pos,
                'end': start_pos + len(''.join(current_entity))
            })
        
        return entities
    
    def batch_predict(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def pretty_print(self, prediction_result):
        """美化打印预测结果"""
        text = prediction_result['text']
        entities = prediction_result['entities']
        
        print(f"文本: {text}")
        print(f"实体: {len(entities)}个")
        
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity['text']} ({entity['type']})")
        
        # 可视化标注
        print("\n标注结果:")
        colored_text = text
        # 按位置倒序插入标记，避免影响位置
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            colored_text = (
                colored_text[:entity['end']] + 
                f"[{entity['type']}]" + 
                colored_text[entity['end']:]
            )
            colored_text = (
                colored_text[:entity['start']] + 
                f"[{entity['text']}]" + 
                colored_text[entity['start']:]
            )
        
        print(colored_text)
        print("-" * 50)


def load_config(config_path):
    """加载配置文件"""
    if config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # 假设是Python配置文件
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.Config
    
    return config


if __name__ == "__main__":
    import os
    
    # 示例用法
    predictor = NERPredictor(Config)
    
    test_texts = [
        "张三在北京的清华大学读书，他来自上海。",
        "2023年5月20日，李四在纽约参加了人工智能会议。",
        "苹果公司首席执行官蒂姆·库克访问了中国北京。"
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        predictor.pretty_print(result)
        print()