# -*- coding: utf-8 -*-

"""
配置参数信息
"""


Config = {
    "model_path": "output",
    "bert_model_path": r"D:/Miniconda3/bert-base-chinese",

    "train_data_path": r"data/sample.json",
    "valid_data_path": r"data/sample.json",

    "input_max_length": 80,
    "output_max_length": 24,
    "embed_size": 768,

    "epoch_num": 20,
    "batch_size": 30,
    "optimizer": "adam",
    "learning_rate": 1e-3,

    "seed": 42,
    "beam_size": 5
}
