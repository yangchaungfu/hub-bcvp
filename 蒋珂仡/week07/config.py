# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",           #训练好的模型权重文件（.pth 或 .bin）会被保存在这个名为 output 的文件夹里
    "data_path": "文本分类练习.csv",  # CSV路径
    "vocab_path": "chars.txt",       #词表
    "model_type": "fast_text",      # 默认模型，main.py中会循环修改它
    "max_length": 50,               # 根据之前分析，50足够
    "hidden_size": 128,             # 隐藏层维度
    "kernel_size": 3,               # CNN卷积核大小
    "num_layers": 2,                # LSTM层数
    "epoch": 5,                     # 训练轮数
    "batch_size": 64,
    "pooling_style": "avg",         # FastText用avg，其他可以用max
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": None,    # 不用BERT，设为None
    "seed": 987
}
