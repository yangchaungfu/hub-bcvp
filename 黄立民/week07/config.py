# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data1/文本分类练习-train.xlsx",
    "valid_data_path": "../data1/文本分类练习-valid.xlsx",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 32,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path":r"E:\pycharmproject\大模型培训-八斗\week7 文本分类问题\pretrain_models\bert-base-chinese",
    "seed": 987
}

