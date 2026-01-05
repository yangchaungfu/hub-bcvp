# -*- coding: utf-8 -*-

"""
参数配置
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_labeled_reviews.csv",
    "valid_data_path": "data/valid_labeled_reviews.csv",
    "test_data_path": "data/test_labeled_reviews.csv",

    "model_type": "bert",
    "pretrain_model_path": r"D:\Miniconda3\bert-base-chinese",

    "vocab_path": "chars.txt",
    "sentence_len": 50,  # 单个文本的长度(单个文本中字词的数量)
    "embed_dim": 36,  # 词向量的维度
    "pooling_style": "avg",

    "num_layers": 3,  # rnn_*/rnn/gru/lstm/的层数
    "kernel_size": 10,  # cnn/*_cnn卷积核大小

    "num_epochs": 12,  # 训练轮数
    "batch_size": 100,  # 每批的训练样本数
    "learning_rate": 1e-3,
    "optimizer": "adam",

    "seed": 987  # 预设的随机数种子，用于确保实验的可重复性
}
