# -*- coding:utf-8 -*-


# 基础配置
Config = {
    "pretrain_model_path": "./data/bert-base-chinese",
    "log_base_path": "./log",
    "model_base_path": "./model",
    "train_data_path": "./data/train.txt",
    "test_data_path": "./data/test.txt",
    "vocab_path": "./data/bert-base-chinese/vocab.txt",
    "model_type": "lstm",
    "label_num": 9,
    "num_epochs": 10,
    "batch_size": 20,
    "hidden_size": 256,
    "learning_rate": 1e-3,
    "max_length": 50,
    "use_crf": True,
    "F1_type": 0     # 计算F1的方式：0为微观，1为宏观
}

# 所有标签
Labels = {
    "O": 0,
    "B-PERSON": 1,
    "I-PERSON": 2,
    "B-LOCATION": 3,
    "I-LOCATION": 4,
    "B-ORGANIZATION": 5,
    "I-ORGANIZATION": 6,
    "B-TIME": 7,
    "I-TIME": 8
}