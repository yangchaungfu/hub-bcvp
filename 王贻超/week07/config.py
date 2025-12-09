# -*- coding: utf-8 -*-

"""
电商评论分类配置参数信息
"""

Config = {
    "model_path": "review_output",
    "train_data_path": "../data/train_review.json",
    "valid_data_path": "../data/valid_review.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 50,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,  # BERT使用更小的学习率
    "pretrain_model_path":r"E:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese",
    "seed": 987,
    "class_num": 2  # 二分类任务
}
