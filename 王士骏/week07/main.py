# -*- coding: utf-8 -*-
import time

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    # 训练集和验证集按照8-2划分，即每5条同类型的样本，4条进入训练集，1条进入验证集
    # 记录样本文本长度
    len_list = []
    with open("文本分类练习.csv", encoding='UTF-8') as f:
        pos = 0
        neg = 0
        with open(config["train_data_path"], mode='w', encoding='UTF-8') as t:
            with open(config["valid_data_path"], mode='w', encoding='UTF-8') as v:
                for index, line in enumerate(f):
                    if index == 1:
                        continue
                    if line[0] == '1':
                        if pos < 4:
                            t.write(line)
                            len_list.append(len(line) - 2)
                            pos += 1
                        else:
                            v.write(line)
                            len_list.append(len(line) - 2)
                            pos = 0
                    elif line[0] == '0':
                        if neg < 4:
                            t.write(line)
                            len_list.append(len(line) - 2)
                            neg += 1
                        else:
                            v.write(line)
                            len_list.append(len(line) - 2)
                            neg = 0
                    else:
                        continue
    # 计算可以覆盖95%的样本数量的文本长度
    config["max_length"] = np.percentile(len_list, 95, method='nearest')

    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_{}.pth".format(Config["model_type"]))
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # main(Config)
    col = ["model_type","learning_rate","hidden_size","epoch","batch_size","pooling_style","optimizer"]

    Config["model_type"] = "stack_gated_cnn"
    result = main(Config)
    print("最后一轮准确率：", result[0], "当前配置：", Config["model_type"],"预测100条耗时：",result[1])
    text = ""
    for i in col:
        text = text + str(Config[i]) + ","
    text = text + str(result[0]) + "," + str(result[1]) + "\n"
    with open("result.csv",mode='a', encoding="utf8") as f:
        f.write(text)

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # # 超参数的网格搜索
    # for model in ["gated_cnn", "lstm", "rnn", "bert"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", 'max']:
    #                     Config["pooling_style"] = pooling_style
    #                     result = main(Config)
    #                     print("最后一轮准确率：", result[0], "当前配置：", Config["model_type"], "预测100条耗时：",
    #                           result[1])
    #                     text = ""
    #                     for i in col:
    #                         text = text + str(Config[i]) + ","
    #                     text = text + str(result[0]) + "," + str(result[1]) + "\n"
    #                     with open("result.csv", mode='a', encoding="utf8") as f:
    #                         f.write(text)
    #
    #
