# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\output",
    "train_data_path": r"F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\data\comments_train.json",
    "valid_data_path": r"F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\data\comments_test.json",
    "vocab_path":r"F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\chars.txt",
    "model_type":"bert",
    "max_length": 71, # 69 + 2
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 6,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path":r"F:\八斗学院\第六周 语言模型\bert-base-chinese",
    "seed": 987
}

