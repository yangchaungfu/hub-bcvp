# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"F:\八斗学院\第九周 序列标注\week9 序列标注问题\ner\model_output",
    "schema_path": r"F:\八斗学院\第九周 序列标注\week9 序列标注问题\ner\ner_data\schema.json",
    "train_data_path": r"F:\八斗学院\第九周 序列标注\week9 序列标注问题\ner\ner_data\train",
    "valid_data_path": r"F:\八斗学院\第九周 序列标注\week9 序列标注问题\ner\ner_data\test",
    # "vocab_path":"chars.txt",
	"vocab_path": r"F:\八斗学院\第六周 语言模型\bert-base-chinese\vocab.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"F:\八斗学院\第六周 语言模型\bert-base-chinese",
	"pretrained_model_path":r"F:\八斗学院\第六周 语言模型\bert-base-chinese",
}