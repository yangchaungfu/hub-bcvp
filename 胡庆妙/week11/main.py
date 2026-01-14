# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import time
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
import loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序。 本例采用 Bert+交叉注意力 实现了SFT式的 Seq2Seq，构建了一个问答模型。

SFT（Supervised Fine-Tuning）式的 Seq2Seq 实现 "续写模型" 到 "问答模型" 的关键。
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载模型
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()  # 是否使用gpu
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载训练数据
    train_data = loader.load_train_data(config["train_data_path"], config)

    # 加载验证类
    evaluator = Evaluator(config, model, logger)

    begin_time = time.time()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 开始训练...")
    epoch = 0
    for epoch in range(config["epoch_num"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            inputs_ids, attention_mask, target_ids = batch_data

            optimizer.zero_grad()  # 梯度归零
            loss = model(inputs_ids, attention_mask, target_ids)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            train_loss.append(loss.item())

            if (index + 1) % int(len(train_data) / 3) == 0:
                logger.info("batch loss %.4f" % loss.item())

        logger.info("epoch average loss: %.4f" % np.mean(train_loss))
        evaluator.eval(epoch)

    spent_time = round((time.time() - begin_time), 2)  # 训练及验证耗时
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 训练结束, 耗时: {spent_time}秒")

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model


if __name__ == "__main__":
    main(Config)
