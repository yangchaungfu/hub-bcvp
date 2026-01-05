# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from loader import load_vocab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    vocab = load_vocab(Config["vocab_path"])  # 加载词表
    model = TorchModel(config)  # 加载模型
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, vocab, shuffle=True)
    # 加载验证类
    evaluator = Evaluator(config, vocab, model, logger)

    # 开始训练
    for epoch in range(config["num_epochs"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data

            optimizer.zero_grad()  # 梯度归零
            loss = model(input_ids, labels)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            train_loss.append(loss.item())

            if (index + 1) % int(len(train_data) / 2) == 0:
                logger.info("batch loss %.4f" % loss.item())

        logger.info("epoch average loss: %.4f" % np.mean(train_loss))
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model


if __name__ == "__main__":
    main(Config)
