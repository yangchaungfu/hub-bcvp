

# -*- coding: utf-8 -*-

"""
配置参数信息 - BERT版本
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/test.txt",
    "vocab_path": "bert-base-chinese",  # 使用BERT的词汇表
    "max_length": 128,                   # BERT通常使用128或256
    "hidden_size": 768,                  # BERT-base的隐藏层大小
    "num_layers": 12,                    # BERT的层数（由预训练模型决定）
    "epoch": 10,                         # 训练轮数
    "batch_size": 32,                    # 批次大小
    "optimizer": "adamw",                # 使用AdamW优化器（适合BERT）
    "learning_rate": 2e-5,               # BERT需要较小的学习率
    "use_crf": True,                     # 是否使用CRF层
    "class_num": 9,                      # 标签类别数
    "bert_path": r"G:\自然语言处理\自然语言处理2025-10-22\bert-base-chinese\bert-base-chinese",    # BERT预训练模型名称
    "device": "cuda",                    # 设备：cuda或cpu
    "dropout_rate": 0.1,                 # dropout率
    "warmup_steps": 100,                 # 学习率预热步数
    "gradient_accumulation_steps": 1,    # 梯度累积步数
    "max_grad_norm": 1.0,                # 梯度裁剪阈值
    "early_stop_patience": 3,            # 早停耐心值
    "save_best_model": True,             # 是否保存最佳模型
    "seed": 42                           # 随机种子
}

