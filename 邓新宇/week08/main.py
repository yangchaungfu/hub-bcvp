# -*- coding: utf-8 -*-
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"], config)

    model = SiameseNetwork(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("使用GPU训练")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)

    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"第{epoch}轮训练开始")
        train_loss = []

        for batch_idx, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            anchor, positive, negative = batch_data

            loss = model(anchor, positive, negative)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                logger.info(f"批次 {batch_idx}/{len(train_data)}, 损失: {loss.item():.4f}")

        avg_loss = np.mean(train_loss)
        logger.info(f"第{epoch}轮平均损失: {avg_loss:.4f}")

        evaluator.eval(epoch)

        model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)

    logger.info("训练完成")
    return


if __name__ == "__main__":
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(Config)