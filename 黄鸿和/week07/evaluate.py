# -*- coding: utf-8 -*-
import torch
from loader import load_data
import time

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果
    
    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc
    import time
    def eval_100(self):
        self.model.eval()

        COUNT_LIMIT = 100
        current_count = 0
        start_time = None
        time_for_100_samples = None

        self.logger.info("--- 开始计时测试：预测前 100 条数据 ---")

        for _, batch_data in enumerate(self.valid_data):
            if time_for_100_samples is not None:
                break
            if torch.cuda.is_available():
                  batch_data = [d.cuda() for d in batch_data]
            input_ids, _ = batch_data
            batch_size = input_ids.size(0)
            with torch.no_grad():
                if start_time is None:
                      start_time = time.time()
                self.model(input_ids)
                current_count += batch_size
                if current_count >= COUNT_LIMIT:
                    time_for_100_samples = time.time() - start_time
                    self.logger.info(
						f"*** 计时完成！预测前 {current_count} 条数据总耗时: {time_for_100_samples:.6f} 秒 ***"
					)
                    break
        if time_for_100_samples is None:
             self.logger.warning(f"警告：数据集总样本数不足 {COUNT_LIMIT} 条。")
        return time_for_100_samples

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
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
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
