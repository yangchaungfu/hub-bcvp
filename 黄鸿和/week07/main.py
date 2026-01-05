# -*- coding: utf-8 -*-

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
            # print('len(train_data)',len(train_data))
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        time_100 = evaluator.eval_100()
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, time_100
import json

RESULTS_JSON_FILE = r"F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\output\model_results.json" 

def save_results_to_json(model_type, accuracy, time_100):
    results = {}
    if os.path.exists(RESULTS_JSON_FILE):
        try:
            with open(RESULTS_JSON_FILE, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"文件 {RESULTS_JSON_FILE} 格式损坏，将覆盖写入。")
            results = {} # 读取失败则重置
        except Exception as e:
            logger.error(f"读取 {RESULTS_JSON_FILE} 时发生错误: {e}")
            results = {}
    results[model_type] = {
        "accuracy": accuracy,
        "time_100": time_100
    }

    try:
        with open(RESULTS_JSON_FILE, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 确保中文显示正常，indent=4 保持格式美观
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"结果已成功保存到 {RESULTS_JSON_FILE}: {model_type} -> {accuracy:.4f}")
    except Exception as e:
        logger.error(f"保存结果到 JSON 文件时发生错误: {e}")

if __name__ == "__main__":
    for model in ["cnn", 'lstm', 'fast_text', 'gru', 'rnn', 'rcnn']:
        Config["model_type"] = model
        last_acc, time_100 = main(Config)
        print("最后一轮准确率：", last_acc, "当前配置：", Config)
        # 调用新函数保存结果
        save_results_to_json(Config["model_type"], last_acc, time_100)
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["cnn", 'bert', 'lstm']:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", 'max']:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)
                        
	


