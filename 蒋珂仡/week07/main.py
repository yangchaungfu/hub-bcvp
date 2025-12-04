# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import pandas as pd
import logging
import time
from sklearn.model_selection import train_test_split
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 随机种子
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config, train_df, valid_df):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据 (build_vocab=True 只在第一次构建词表)
    # 只要 vocab_path 指向的文件存在，loader 就会读取，不用每次都 build
    train_data = load_data(train_df, config, build_vocab=True)

    # 加载模型
    model = TorchModel(config)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger, valid_df)  # 注意：这里修改了Evaluator初始化

    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        logger.info("epoch %d average loss: %f" % (epoch, np.mean(train_loss)))
        acc = evaluator.eval(epoch)

    # 训练结束后进行测速 (预测100条)
    speed_test_time = evaluator.speed_test()

    return acc, speed_test_time


if __name__ == "__main__":
    # 1. 读取 CSV 并划分数据集
    logger.info("正在读取数据...")
    df = pd.read_csv(Config["data_path"])
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=Config["seed"])
    logger.info(f"数据加载完成。训练集: {len(train_df)}, 验证集: {len(valid_df)}")

    # 2. 定义要对比的模型列表
    models_to_run = ["fast_text", "cnn", "lstm"]
    results = []

    # 3. 循环训练
    for model_name in models_to_run:
        logger.info(f"-" * 20)
        logger.info(f"开始训练模型: {model_name}")
        Config["model_type"] = model_name

        # FastText使用avg pooling，其他通常使用max
        if model_name == "fast_text":
            Config["pooling_style"] = "avg"
        else:
            Config["pooling_style"] = "max"

        acc, infer_time = main(Config, train_df, valid_df)

        results.append({
            "Model": model_name,
            "Accuracy": f"{acc:.4f}",
            "Time(100 samples)": f"{infer_time:.2f} ms"
        })

    # 4. 输出最终表格
    print("\n" + "=" * 50)
    print("最终实验结果汇总")
    print("=" * 50)
    result_df = pd.DataFrame(results)
    print(result_df)
