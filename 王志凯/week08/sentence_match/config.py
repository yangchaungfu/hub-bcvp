# -*- coding:utf-8 -*-

Config = {
    "pretrain_model_path": "./data/bert-base-chinese",
    "log_base_path": "./log",
    "model_base_path": "./model",
    "train_data_path": "./data/train.json",
    "valid_data_path": "./data/valid.json",
    "schema_path": "./data/schema.json",
    "model_name": "bert",
    "model_type": "bert",
    "train_type": "Siam",         # "Siam"：孪生网络，"Triply"：三元组
    "matching_type": "cosine",    # 从encoder层输出之后的matching layer。"concat"：拼接， "cosine"：直接计算余弦距离
    "concat_type": 0,             # matching_type为"concat"时生效，0:(u,v)   1:(u,v,|u-v|)   2:(u,v,u*v)
    "margin": 0.1,                # 正负样本阈值系数（建议0.1 - 0.5）,train_type="Triply"时生效
    "num_epochs": 10,
    "hidden_size": 256,
    "out_channels": 128,
    "epoch_size": 200,            # 每轮训练样本总数
    "learning_rate": 1e-3,
    "batch_size": 20,
    "max_length": 30,
    "kernel_size": 3
}