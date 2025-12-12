# -*- coding:utf-8 -*-

import torch
import os
import torch.nn as nn
from loader import load_data
from logHandler import logger
from collections import defaultdict
from transformers import BertTokenizer
logger = logger(os.path.basename(__file__))


class Evaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.valid_data = load_data(config, isTrain=False)
        self.index_question_label = defaultdict(list)
        self.faq2vec()

    # 将知识库中所有的问题转化为向量，并与标准问建立映射
    def faq2vec(self):
        # 标准问对应的所有问题的序列
        q_vec_dict = self.valid_data.dataset.q_vec_dict
        # 标准问对应的所有问题的字符串
        q_sent_dict = self.valid_data.dataset.q_sent_dict
        # 如果是批量测试bert模型，只能将批量数据一次性进行转化，否则送入bert模型的数据格式很难处理，
        # 因为每条数据都包含input_ids，attention_mask，token_type_ids
        if self.config["model_type"] == "bert":
            tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
            q_sent = list(q_sent_dict.values())
            q_vecs = tokenizer(q_sent, truncation=True, padding="max_length", max_length=self.config["max_length"],
                               return_tensors='pt')
            q_labels = q_sent_dict.keys()
        else:
            # list(q_vec_dict.values())是tensor列表，使用torch.stack可以将其堆叠成一个tensor
            q_vecs = torch.stack(list(q_vec_dict.values()))
            q_labels = q_vec_dict.keys()
        # 将所有问题向量化
        self.faq_vecs = self.model(q_vecs)
        for index, (vec, label) in enumerate(zip(self.faq_vecs, q_labels)):
            # 用索引对应问题和标准问label{0：[seq, label], 1: [seq, label]...}
            label = int(label.split("_")[0])
            self.index_question_label[index] = [vec, label]

    def predict(self):
        logger.info(f"开始对模型进行预测")
        print(f"开始对模型进行预测")
        correct = 0
        wrong = 0
        self.model.eval()
        with torch.no_grad():
            for batch_x, labels in self.valid_data:
                input_vecs = self.model(batch_x)
                for input_vec, label in zip(input_vecs, labels):
                    input_norm = nn.functional.normalize(input_vec, dim=-1).unsqueeze(0)
                    faq_vecs_norm = nn.functional.normalize(self.faq_vecs, dim=-1)
                    # 输入文本与知识库中所有问题的相似度 res.shape = (1, n)
                    res = input_norm @ faq_vecs_norm.T
                    # 取出最大值的索引即为相似度最高的问题向量在self.faq_vecs中的位置
                    hit_idx = torch.argmax(res.squeeze())
                    hit_label = self.index_question_label[int(hit_idx)][1]
                    print(hit_label)
                    if int(label) == int(hit_label):
                        correct += 1
                    else:
                        wrong += 1
        logger.info(f"预测总数：{correct + wrong}， 准确率为：{correct / (correct + wrong):.4%}")
        print(f"预测总数：{correct + wrong}， 准确率为：{correct / (correct + wrong):.4%}")
