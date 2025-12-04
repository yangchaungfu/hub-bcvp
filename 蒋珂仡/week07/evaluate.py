# -*- coding: utf-8 -*-
import torch
import time
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger, valid_df):
        self.config = config
        self.model = model
        self.logger = logger
        # 这里改为接收 valid_df
        self.valid_data = load_data(valid_df, config, shuffle=False, build_vocab=False)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        return correct / (correct + wrong)

    def speed_test(self):
        # 生成100条随机数据用于测速
        self.model.eval()
        sample_input = torch.randint(0, self.config["vocab_size"], (100, self.config["max_length"]))
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()

        # 预热
        self.model(sample_input)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # 测10次取平均
                self.model(sample_input)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10 * 1000  # ms
        self.logger.info(f"预测100条耗时: {avg_time:.2f} ms")
        return avg_time
