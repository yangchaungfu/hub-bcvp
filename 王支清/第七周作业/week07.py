# -*- coding: utf-8 -*-
"""
文本分类模型实现
整合了配置、数据加载、模型定义、训练和评估功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
import random
import time
import logging
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ==================== 配置参数 ====================
Config = {
    "model_path": "output",
    "data_path": "文本分类练习.csv",
    "vocab_path": "chars.txt",
    "model_type": "fast_text",
    "max_length": 50,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 64,
    "pooling_style": "avg",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": None,
    "seed": 987,
    "class_num": 2
}


# ==================== 数据加载模块 ====================
class DataGenerator(Dataset):
    def __init__(self, data, config, build_vocab=False):
        self.config = config
        self.data = data

        if build_vocab:
            self.build_vocab()
        else:
            self.vocab = self.load_vocab(config["vocab_path"])

        self.config["vocab_size"] = len(self.vocab)
        self.dataset = []
        self.load()

    def build_vocab(self):
        all_text = "".join([str(text) for text in self.data['review']])
        counts = Counter(all_text)
        vocab = {char: i + 2 for i, (char, _) in enumerate(counts.most_common(4000))}
        vocab['[PAD]'] = 0
        vocab['[UNK]'] = 1
        self.vocab = vocab

        with open(self.config["vocab_path"], 'w', encoding='utf-8') as f:
            for key, value in self.vocab.items():
                f.write(json.dumps([key, value]) + "\n")

    def load_vocab(self, vocab_path):
        token_dict = {}
        try:
            with open(vocab_path, encoding="utf8") as f:
                for line in f:
                    pair = json.loads(line)
                    token_dict[pair[0]] = pair[1]
        except FileNotFoundError:
            return {}
        return token_dict

    def load(self):
        for index, row in self.data.iterrows():
            review = str(row['review'])
            label = int(row['label'])

            input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.dataset.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def load_data(data_frame, config, shuffle=True, build_vocab=False):
    dg = DataGenerator(data_frame, config, build_vocab=build_vocab)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


# ==================== 模型定义 ====================
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                                   batch_first=True, bidirectional=True)
            hidden_size = hidden_size * 2
        elif model_type == "cnn":
            self.encoder = CNN(config)

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        x = self.embedding(x)
        x = self.encoder(x)

        if isinstance(x, tuple):
            x = x[0]

        if self.pooling_style == "max":
            pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            pooling_layer = nn.AvgPool1d(x.shape[1])

        x = x.transpose(1, 2)
        x = pooling_layer(x).squeeze()
        predict = self.classify(x)

        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")


# ==================== 评估模块 ====================
class Evaluator:
    def __init__(self, config, model, logger, valid_df):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(valid_df, config, shuffle=False, build_vocab=False)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果：")
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}

        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data

            with torch.no_grad():
                pred_results = self.model(input_ids)
            self.write_stats(labels, pred_results)

        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong

        self.logger.info(f"预测集合条目总量：{total}")
        self.logger.info(f"预测正确条目：{correct}，预测错误条目：{wrong}")
        self.logger.info(f"预测准确率：{correct / total:.4f}")

        return correct / total

    def speed_test(self):
        self.model.eval()
        sample_input = torch.randint(0, self.config["vocab_size"],
                                     (100, self.config["max_length"]))
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()

        self.model(sample_input)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                self.model(sample_input)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10 * 1000
        self.logger.info(f"预测100条耗时: {avg_time:.2f} ms")
        return avg_time


# ==================== 训练模块 ====================
def main(config, train_df, valid_df, logger):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(train_df, config, build_vocab=True)
    model = TorchModel(config)

    if torch.cuda.is_available():
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger, valid_df)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        logger.info(f"epoch {epoch} average loss: {np.mean(train_loss):.4f}")
        acc = evaluator.eval(epoch)

    speed_test_time = evaluator.speed_test()
    return acc, speed_test_time


# ==================== 主程序 ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    seed = Config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("正在读取数据...")
    df = pd.read_csv(Config["data_path"])
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=Config["seed"])
    logger.info(f"数据加载完成。训练集: {len(train_df)}, 验证集: {len(valid_df)}")

    models_to_run = ["fast_text", "cnn", "lstm"]
    results = []

    for model_name in models_to_run:
        logger.info("-" * 40)
        logger.info(f"开始训练模型: {model_name}")
        Config["model_type"] = model_name

        if model_name == "fast_text":
            Config["pooling_style"] = "avg"
        else:
            Config["pooling_style"] = "max"

        acc, infer_time = main(Config, train_df, valid_df, logger)

        results.append({
            "Model": model_name,
            "Accuracy": f"{acc:.4f}",
            "Time(100 samples)": f"{infer_time:.2f} ms"
        })

    print("\n" + "=" * 50)
    print("最终实验结果汇总")
    print("=" * 50)
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
