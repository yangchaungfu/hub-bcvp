# -*- coding: utf-8 -*-

"""
main文件主要任务：参数拼接、模型训练、结果对比和输出模型
"""

import torch
import time
import numpy as np
import os
from config import Config
from logHandler import logger
from loader import load_train_data, load_vocab
from model import TorchModel
from evaluator import Evaluator

logger = logger()
gpu_usable = torch.cuda.is_available()


def main(config):
    # 加载数据、模型和优化器
    batch_data = load_train_data(config)
    model = TorchModel(config)
    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if gpu_usable:
        logger.info(f"====GPU可用====")
        model = model.cuda()
    # 开始训练
    logger.info(f"====开始训练{config["model_name"]}模型====")
    print(f"====开始训练{config["model_name"]}模型====")
    start = time.time()
    model.train()
    for i in range(config["epoch"]):
        watch_loss = []
        for index, data in enumerate(batch_data):
            if gpu_usable:
                data = [d.cuda() for d in data]
            batch_x, batch_y = data
            loss = model(batch_x, batch_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        # 记录loss信息
        logger.info(f"第{i+1}轮训练结束，该轮平均loss值为：{np.mean(watch_loss)}")
        print(f"第{i+1}轮训练结束，该轮平均loss值为：{np.mean(watch_loss)}")
    # 记录训练耗时
    execution_time = time.time() - start
    config["execution_time"] = execution_time
    # 使用测试集预测，记录预测速率、准确率
    evaluator = Evaluator(config, model)
    evaluator.predict()
    # 保存模型
    if config["save_model"]:
        model_name = config["model_name"] + ".pth"
        model_path = os.path.join(config["model_path"], model_name)
        torch.save(model.state_dict(), model_path)
    # 记录模型参数
    logger.info(f"模型参数：{config}")
    # print(f"模型参数：{config}")


if __name__ == "__main__":
    vocab_size = load_vocab(Config["vocab_path"])
    Config["vocab_size"] = vocab_size
    # 使用不同的模型和超参数来训练，对比结果
    model_type_list = ["bert"]
    # model_type_list = ["LSTM", "RNN", "CNN", "bert"]
    # model_type_list = ["fastText", "RNN", "CNN", "TextCNN", "RCNN", "LSTM", "bert", "bertRNN"]
    num_layers_list = [1, 3]
    bidirectional_list = [True, False]
    lr_list = [1e-3, 1e-4]
    batch_size_list = [20, 40]
    hidden_size_list = [256, 512]
    out_channels_list = [64, 128]
    pooling_type_list = ["max", "avg"]
    for model in model_type_list:
        count = 0
        Config["model_type"] = model
        # 如果是普通的fastText模型，则只比较不同学习率和batch_size下的模型
        if model == "fastText":
            for lr in lr_list:
                Config["learning_rate"] = lr
                for batch_size in batch_size_list:
                    count += 1
                    Config["model_name"] = model + "_" + str(count)
                    Config["batch_size"] = batch_size
                    main(Config)
        # CNN，只比较feature_dim、hidden_size、out_channels
        elif model == "CNN":
            for hidden_size in hidden_size_list:
                Config["hidden_size"] = hidden_size
                for out_channels in out_channels_list:
                    Config["out_channels"] = out_channels
                    for pooling_type in pooling_type_list:
                        count += 1
                        Config["model_name"] = model + "_" + str(count)
                        Config["pooling_type"] = pooling_type
                        main(Config)
        # TextCNN，只比较堆叠层数和out_channels
        elif model == "TextCNN":
            for num_layers in num_layers_list:
                Config["num_layers"] = num_layers
                for out_channels in out_channels_list:
                    count += 1
                    Config["model_name"] = model + "_" + str(count)
                    Config["out_channels"] = out_channels
                    main(Config)
        # bert，只比较num_layers
        elif model == "bert":
            for num_layers in num_layers_list:
                count += 1
                Config["model_name"] = model + "_" + str(count)
                Config["num_layers"] = num_layers
                main(Config)
        # 比较不同feature_dim、hidden_size和bidirectional下的模型
        elif model in ["RNN", "RCNN", "LSTM", "bertRNN"]:
            for hidden_size in hidden_size_list:
                Config["hidden_size"] = hidden_size
                for bidirectional in bidirectional_list:
                    count += 1
                    Config["model_name"] = model + "_" + str(count)
                    Config["bidirectional"] = bidirectional
                    main(Config)

