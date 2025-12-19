# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "bert_model_path": r"D:/Miniconda3/bert-base-chinese",
    "vocab_path": "D:/Miniconda3/bert-base-chinese/vocab.txt",

    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/valid.txt",
    "test_data_path": "ner_data/test.txt",

    "sentence_len": 90,
    "embed_size": 768,
    "class_num": 9,

    "num_epochs": 15,  # 训练轮数
    "batch_size": 16,  # 每批的训练样本数
    "learning_rate": 0.0004,
    "optimizer": "adam",
}
