#  -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import time
from datetime import datetime

"""
模型训练主程序
"""

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)  # 设random模块的随机种子
np.random.seed(seed)  # 设置numpy模块的随机种子
torch.manual_seed(seed)  # 设置PyTorch的随机种子, 影响PyTorch的CPU随机数生成
torch.cuda.manual_seed_all(seed)  # 设置所有可用GPU的随机种子, 确保在多GPU环境下的可重复性


def main(config):
    # 创建模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据(必须先加载训练数据，再加载模型，因为加载模型时依赖vocab_size)
    train_data = load_data(config["train_data_path"], config)

    model = TorchModel(config)  # 加载模型
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    evaluator = Evaluator(config, model, logger)  # 加载验证类
    optimizer = choose_optimizer(config, model)  # 加载优化器

    # 训练
    for epoch in range(config["num_epochs"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()  # 梯度归零
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况

            loss = model(input_ids, labels)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = round(evaluator.eval(epoch), 4)

    model_path = os.path.join(config["model_path"], f"{config['model_type']}.pth")
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc


def save_train_log(filename, config, last_epoch_acc, spent_time):
    # 创建文件时写入标题栏
    if not os.path.exists(filename):
        # Python会将括号内的多行字符串自动连接
        title = ("model_type, batch_size, learning_rate, sentence_len, "
                 "embed_dim, pooling_style, num_layers, kernel_size, "
                 "num_epochs, 最后一轮准确率, 训练及验证的耗时(秒)")
        with open(filename, 'a', encoding='gbk') as f:
            f.write(title + '\n')

    # 写入训练参数及训练效果
    content = (f"{config['model_type']}, {config['batch_size']}, {config['learning_rate']}, {config['sentence_len']}, "
               f"{config['embed_dim']}, {config['pooling_style']}, {config['num_layers']}, {config['kernel_size']}, "
               f"{config['num_epochs']}, {last_epoch_acc}, {spent_time}")
    with open(filename, 'a', encoding='gbk') as f:
        f.write(content + '\n')


def train_and_record(config, train_log_fname):
    begin_time = time.time()
    last_epoch_acc = main(Config)  # 训练及验证
    spent_time = round((time.time() - begin_time), 2)  # 训练及验证耗时

    print(f"训练及验证耗时：{spent_time} 秒, 最后一轮准确率：{last_epoch_acc}, 当前配置：\n\t {Config}")
    save_train_log(train_log_fname, config, last_epoch_acc, spent_time)


if __name__ == "__main__":
    # main(Config)

    # 超参数的网格搜索  对比所有模型
    # for model in ["fast_text", "rnn", 'gru', 'lstm', "cnn", "gated_cnn",
    #               "stack_gated_cnn", "rnn_cnn", "bert", "bert_cnn"]:

    train_log_filename = os.path.join(Config["model_path"],
                                      f"train_log_{datetime.now().strftime('%Y%m%d%H%M')}.csv")
    for model_type in ["fast_text", "rnn", "gru", "cnn", "bert"]:
        Config["model_type"] = model_type
        for batch_size in [64, 128]:
            Config["batch_size"] = batch_size
            for learning_rate in [1e-3, 2e-3]:
                Config["learning_rate"] = learning_rate
                for sentence_len in [50, 80]:
                    Config["sentence_len"] = sentence_len
                    for embed_dim in [24, 48]:
                        Config["embed_dim"] = embed_dim

                        # 由于模型bert没有用到 pooling_style, num_layers, kernel_size, 所以不搜索这几个参数
                        if model_type in ["bert"]:
                            Config["pooling_style"] = '-'
                            Config["num_layers"] = '-'
                            Config["kernel_size"] = '-'
                            train_and_record(Config, train_log_filename)

                        # 处理非'bert'的
                        else:
                            for pooling_style in ["avg", "max"]:
                                Config["pooling_style"] = pooling_style

                                # 由于模型fast_text没有用到 num_layers, kernel_size, 所以不搜索这几个参数
                                if model_type in ["fast_text"]:
                                    Config["num_layers"] = '-'
                                    Config["kernel_size"] = '-'
                                    train_and_record(Config, train_log_filename)

                                # 用到参数num_layers 的模型
                                elif model_type in ["rnn", "gru", "lstm"]:
                                    for num_layers in [1, 3]:
                                        Config["kernel_size"] = '-'
                                        Config["num_layers"] = num_layers
                                        train_and_record(Config, train_log_filename)

                                # 用到参数kernel_size 的模型
                                elif model_type in ["cnn", "gated_cnn", "rnn_cnn", "bert_cnn"]:
                                    for kernel_size in [16, 32]:
                                        Config["num_layers"] = '-'
                                        Config["kernel_size"] = kernel_size
                                        train_and_record(Config, train_log_filename)

                                elif model_type in ["stack_gated_cnn"]:
                                    for num_layers in [1, 3]:
                                        Config["num_layers"] = num_layers
                                        for kernel_size in [16, 32]:
                                            Config["kernel_size"] = kernel_size
                                            train_and_record(Config, train_log_filename)
