Config = {
    # 输出
    "model_path": "model_output",

    # 数据与标注
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",

    # 训练参数
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adamw",

    # BERT 常用学习率量级
    "learning_rate": 2e-5,

    # 序列长度
    "max_length": 128,

    # CRF
    "use_crf": True,

    # 类别数
    "class_num": 9,

    # BERT 路径
    "bert_path": r"E:\pretrain_models\bert-base-chinese-2",

    # dropout
    "dropout": 0.1,
}
