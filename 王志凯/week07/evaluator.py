# -*- coding: utf-8 -*-

"""
evaluator任务：对测试集进行测试、比较不同模型不同参数训练结果、将config数据进行输出
"""

import torch
import time
import numpy as np
from logHandler import logger
from loader import load_valid_data
logger = logger()

class Evaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.efficiency = 0
        self.correct_percent = 0

    def predict(self):
        print(f"===使用模型{self.config["model_name"]}开始预测===")
        batch_data = load_valid_data(self.config)
        # 平均每百条耗时（最后一个batch可能不足100条，要去掉）
        execution_time_avg = []
        correct = 0
        wrong = 0
        self.model.eval()
        for batch_x, batch_y in batch_data:
            start = time.time()
            with torch.no_grad():
                y_pred = self.model(batch_x)
                for y_p, y in zip(y_pred, batch_y):
                    if y_p[int(y[0])] > 0.5:
                        correct += 1
                    else:
                        # print(f"预测错误！预测值为：{y_p}, 真实值为：{y}")
                        wrong += 1
            # 每个batch耗时
            execution_time = time.time() - start
            execution_time_avg.append(execution_time)
        self.efficiency = f"{np.mean(execution_time_avg[:-1]):.6f}"
        print(f"每百条预测耗时：{self.efficiency}")
        # 计算正确率
        self.correct_percent = f"{correct / (correct + wrong):.4%}"
        print(f"预测准确率：{self.correct_percent}")
        self.model_config()

    # 整理出模型参数和训练结果
    def model_config(self):
        model_test_info = dict()
        model_test_info["model"] = self.config["model_name"]
        model_test_info["learning_rate"] = self.config["learning_rate"]
        model_test_info["batch_size"] = self.config["batch_size"]
        model_test_info["hidden_size"] = self.config["hidden_size"]
        model_test_info["out_channels"] = self.config["out_channels"]
        model_test_info["num_layers"] = self.config["num_layers"]
        model_test_info["bidirectional"] = self.config["bidirectional"]
        model_test_info["pooling_type"] = self.config["pooling_type"]
        model_test_info["efficiency"] = self.efficiency
        model_test_info["correct_percent"] = self.correct_percent
        logger.info(model_test_info)





