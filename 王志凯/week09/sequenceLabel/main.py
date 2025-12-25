# -*- coding:utf-8 -*-


"""
使用 Bert + CRF训练序列标注模型
"""
import torch
import numpy as np
import os
from transformers import BertConfig
from config import Config
from model import SequenceLabelModel
from loader import load_data
from evaluate import Evaluator
from logHandler import logger
bertConfig = BertConfig.from_pretrained(Config["pretrain_model_path"])
# 初始化日志对象，传入当前文件名，打印的日志会保存于 “当前文件名.log” 文件中
log = logger(os.path.basename(__file__))
cuda = torch.cuda.is_available()


def main(config):
    model = SequenceLabelModel(config)
    if cuda:
        log.info("cuda is available, start training by GPU...")
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_data = load_data(config, config["train_data_path"])

    model.train()
    for epoch in range(config["num_epochs"]):
        log.info(f"===开始第{epoch + 1}轮训练===")
        print(f"===开始第{epoch + 1}轮训练===")
        watch_loss = []
        count = 0
        for index, batch_data in enumerate(train_data):
            if cuda:
                batch_data = [d.cuda() for d in batch_data]
            batch_x, batch_y = batch_data
            optim.zero_grad()
            loss = model(batch_x, batch_y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
            # 每训练20%检查一次
            if index % int(len(train_data) / 5) == 0 and index > 0:
                count += 1
                log.info(f"训练进度：{count / 5:.0%}")
                print(f"训练进度：{count / 5:.0%}")
        log.info(f"第{epoch + 1}轮训练结束，平均loss为：{np.mean(watch_loss):.6f}")
        print(f"第{epoch + 1}轮训练结束，平均loss为：{np.mean(watch_loss):.6f}")
        # 对每轮的训练结果进行测试
        evaluator = Evaluator(config, model)
        evaluator.predict()
    # 保存模型
    model_base_path = config["model_base_path"]
    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
    save_path = os.path.join(model_base_path, config["model_type"] + ".pth")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    Config["vocab_size"] = bertConfig.vocab_size
    log.info(f"config:{Config}")
    main(Config)