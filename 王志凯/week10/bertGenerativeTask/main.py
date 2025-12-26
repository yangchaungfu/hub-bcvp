# -*- coding:utf-8 -*-

import os
import torch
import numpy as np
from transformers import BertConfig
from logHandler import logger
from config import Config
from model import BertGenerativeModel
from loader import load_data
from evaluate import Evaluator
bertConfig = BertConfig.from_pretrained(Config["bert_path"])
log = logger(__file__)
cuda = torch.cuda.is_available()


def main(config):
    # 加载模型、优化器、训练数据
    model = BertGenerativeModel(config)
    if cuda:
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
            loss = model(batch_x, batch_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
            if index % int((len(train_data) / 10)) == 0 and index > 0:
                count += 1
                log.info(f"训练进度：{count / 10:.0%}")
                print(f"训练进度：{count / 10:.0%}")
        log.info(f"第{epoch + 1}轮训练结束，平均loss：{np.mean(watch_loss):.6f}")
        print(f"第{epoch + 1}轮训练结束，平均loss：{np.mean(watch_loss):.6f}")
        # 对该轮结果进行测试
        sentence = "山顶上隐约"
        evaluator = Evaluator(config, model, sentence)
        evaluator.eval()
    # 保存模型
    model_base_path = config["model_base_path"]
    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
    model_path = os.path.join(model_base_path, config["model_type"] + ".pth")
    torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    Config["vocab_size"] = bertConfig.vocab_size
    main(Config)