# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "corpus_path": "sample_data.json",  # 语料路径改为sample_data.json
    "vocab_path":"vocab.txt",
    "hidden_size": 768,     # BERT模型隐藏层维度（bert-base-chinese为768）
    "learning_rate": 1e-3,  # 学习率
    "optimizer": "adam",    # 优化器,adam或sgd
    "epoch_num": 20,        # 训练轮数
    "batch_size": 64,       # 每次训练样本个数
    "train_sample": 50000,  # 每轮训练总共训练的样本总数
    "char_dim": 768,        # 每个字的维度（BERT输出维度）
    "max_input_len": 50,    # 输入序列最大长度(title)
    "max_output_len": 200,  # 输出序列最大长度(content)
    "text_length": 50,      # 生成文本长度
    "bert_path": "bert-base-chinese"  # BERT模型路径（使用huggingface默认路径）
}

