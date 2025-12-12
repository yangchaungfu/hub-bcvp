# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model_triplet_loss import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader_triplet_loss import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = SiameseNetwork(config)

    # 标识是否使用 GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU 可用，迁移模型至 GPU")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("Epoch %d 开始" % epoch)
        train_loss = []

        # 遍历每个 batch
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            # 将数据迁移到 GPU
            if cuda_flag:
                sentence1, sentence2, sentence3, labels = [d.cuda() for d in batch_data]
            else:
                sentence1, sentence2, sentence3, labels = batch_data

            # 计算三元组损失
            loss = model(sentence1, sentence2, sentence3)  # 使用三元组损失进行训练

            train_loss.append(loss.item())

            # 反向传播并优化
            loss.backward()
            optimizer.step()

        # 输出当前 epoch 的平均损失
        logger.info("Epoch %d 平均损失: %f" % (epoch, np.mean(train_loss)))

        # 每个 epoch 结束后进行评估
        evaluator.eval(epoch)

    # 保存模型
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    logger.info("模型保存至 %s" % model_path)


if __name__ == "__main__":
    main(Config)
