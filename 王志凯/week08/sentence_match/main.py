# -*- coding:utf-8 -*-

"""
使用sentence-bert + Triplet loss 实现表示型文本匹配
"""

import torch
import numpy as np
import copy
import os
from transformers import BertConfig
from config import Config
from logHandler import logger
from model import MatchModel
from evaluator import Evaluator
from csvHandler import CSVHandler
from loader import load_data
bertConfig = BertConfig.from_pretrained(Config["pretrain_model_path"])
# 初始化日志对象，传入当前文件名，打印的日志会保存于 “当前文件名.log” 文件中
logger = logger(os.path.basename(__file__))


def main(config):
    logger.info(f"准备训练模型{config["model_name"]}，模型参数为：{config}")
    model = MatchModel(config)
    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_data = load_data(config)
    model.train()
    logger.info(f"====开始训练模型{config["model_name"]}====")
    print(f"====开始训练模型{config["model_name"]}====")
    for i in range(config["num_epochs"]):
        watch_loss = []
        for batch_data in train_data:
            if config["train_type"] == "Siam":
                batch_sent1, batch_sent2, target = batch_data
                loss = model(batch_sent1, batch_sent2, target)
            else:
                a, p, n = batch_data
                loss = model(a, p, n)
            optim.zero_grad()
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        logger.info(f"第{i+1}轮训练，平均loss值为：{np.mean(watch_loss):.6f}")
        print(f"第{i+1}轮训练，平均loss值为：{np.mean(watch_loss):.6f}")
    # 模型预测
    evaluator = Evaluator(config, model)
    evaluator.predict()
    # 保存模型
    model_name = config["model_name"] + ".pth"
    model_path = os.path.join(config["model_base_path"], model_name)
    torch.save(model.state_dict(), model_path)



if __name__ == '__main__':
    Config["vocab_size"] = bertConfig.vocab_size
    config = copy.deepcopy(Config)
    model_list = ["bert"]
    # model_list = ["LSTM", "TextCNN", "bert"]
    train_type_list = ["Siam", "Triplet"]
    matching_type_list = ["concat", "cosine"]
    hidden_size_list = [128, 256]
    out_channels_list = [128, 256]
    concat_type = [0, 1]
    margin_list = [0.1, 0.2]
    # 使用不同的方式训练模型
    for model_type in model_list:
        count = 0
        # 每次使用新模型训练时都要重置config
        Config = config
        Config["model_type"] = model_type
        if model_type in ["LSTM", "TextCNN"]:
            for hidden_size in hidden_size_list:
                Config["hidden_size"] = hidden_size
                if model_type == "TextCNN":
                    # 如果是TextCNN，使用不同的out_channels
                    for out_channels in out_channels_list:
                        Config["out_channels"] = out_channels
                        count += 1
                        Config["model_name"] = model_type + "_" + str(count)
                        main(Config)
                else:
                    count += 1
                    Config["model_name"] = model_type + "_" + str(count)
                    main(Config)
        else:
            # 判断使用孪生网络还是三元组网络
            for train_type in train_type_list:
                Config["train_type"] = train_type
                if train_type == "Siam":
                    # 如果是孪生网络，对比拼接和余弦值
                    for matching_type in matching_type_list:
                        Config["matching_type"] = matching_type
                        if matching_type == "concat":
                            # 不同的拼接方式
                            for concat_type in concat_type:
                                Config["concat_type"] = concat_type
                                count += 1
                                Config["model_name"] = model_type + "_" + str(count)
                                main(Config)
                        else:
                            count += 1
                            Config["model_name"] = model_type + "_" + str(count)
                            main(Config)
                else:
                    # 如果是三元组网络，则需要设置正负样本阈值
                    for margin in margin_list:
                        Config["margin"] = margin
                        count += 1
                        Config["model_name"] = model_type + "_" + str(count)
                        main(Config)

    # 模型参数都打印到日志中，需要将参数从main.log中取出来并写入csv文件
    log_name = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    log_path = os.path.join(Config["log_base_path"], log_name)
    handler = CSVHandler(log_path, "./model_compare.csv")
    # 写入csv（simplify=True：对字段进行加工，只将必要字段写入csv）
    handler.write2csvByDicts(simplify=True)