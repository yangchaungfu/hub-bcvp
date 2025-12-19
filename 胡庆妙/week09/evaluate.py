# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果验证
"""


class Evaluator:
    def __init__(self, config, vocab, model, logger):
        self.config = config
        self.vocab = vocab
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, vocab, shuffle=False)
        self.batch_size = config["batch_size"]
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}

    def eval(self, epoch):
        self.logger.info("开始验证 第%d轮 的训练效果：" % epoch)
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.batch_size: (index + 1) * self.batch_size]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            input_ids, labels = batch_data
            with torch.no_grad():
                pred_labels = self.model(input_ids)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(pred_labels, labels, sentences)
        self.show_stats()
        return

    def write_stats(self, pred_labels, true_labels, sentences):
        """
        Args:
            pred_labels: shape: [batch_size, sen_len]
            true_labels: shape: [batch_size, sen_len]
            sentences: shape: [batch_size]
        """
        assert len(pred_labels) == len(true_labels) == len(sentences)
        for pred_label, true_label, sentence in zip(pred_labels, true_labels, sentences):
            pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            pred_entities = self.decode_label(sentence, pred_label)
            true_entities = self.decode_label(sentence, true_label)

            # print("pred_entities", pred_entities)
            # print("true_entities", true_entities)
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        # F1分数是 精确率（Precision）和 召回率（Recall）的调和平均数： F1 = 2 × (Precision × Recall) / (Precision + Recall)
        # 按类别计算F1，再取平均值作为 Macro-F1
        f1_scores = []  # 每个类别的F1
        for key in ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            f1 = (2 * precision * recall) / (precision + recall + 1e-5)
            f1_scores.append(f1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, f1))
        self.logger.info("Macro-F1(宏观F1): %f" % np.mean(f1_scores))  # 取各类别F1的平均值作为 Macro-F1

        # 汇总所有TP/FP/FN后，计算 Micro-F1
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]])
        tot_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]])
        tot_samp = sum([self.stats_dict[key]["样本实体数"] for key in ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]])
        micro_precision = correct_pred / (tot_pred + 1e-5)
        micro_recall = correct_pred / (tot_samp + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1(微观F1): %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''

    @staticmethod
    def decode_label(sentence, labels):
        """
        Args:
            sentence: 如： "他是彭德怀"
            labels: 如：[8, 8, 2, 6, 6]
        Returns:
            entities: 如：{"PERSON": ["彭德怀", ...]}
        """
        labels = "".join([str(x) for x in labels])
        results = defaultdict(list)
        # 匹配以"0"开头，后跟一个或多个"4"的序列， 以这个序列对应的切分段作为地址实体
        for segment in re.finditer("(04+)", labels):
            s, e = segment.span()  # 匹配到的子字符串的起始和结束位置
            results["LOCATION"].append(sentence[s:e])
        # 匹配以"1"开头，后跟一个或多个"5"的序列， 以这个序列对应的切分段作为机构实体
        for segment in re.finditer("(15+)", labels):
            s, e = segment.span()
            results["ORGANIZATION"].append(sentence[s:e])
        # 匹配以"2"开头，后跟一个或多个"6"的序列， 以这个序列对应的切分段作为人名实体
        for segment in re.finditer("(26+)", labels):
            s, e = segment.span()
            results["PERSON"].append(sentence[s:e])
        # 匹配以"3"开头，后跟一个或多个"7"的序列， 以这个序列对应的切分段作为时间实体
        for segment in re.finditer("(37+)", labels):
            s, e = segment.span()
            results["TIME"].append(sentence[s:e])
        return results
