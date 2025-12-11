# -*- coding: utf-8 -*-
import torch
from loader_triplet_loss import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    # 将知识库中的问题向量化，为匹配做准备
    # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        # 为了生成三元组，我们需要确保加载的每个问题对包含 anchor, positive 和 negative
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)

        with torch.no_grad():
            # 为了生成三元组，问题集需要每三个句子一起形成三元组（anchor, positive, negative）
            triplets = []  # 这个会存放每个三元组：三个句子构成一个三元组
            for i in range(0, len(self.question_ids), 3):
                if i + 2 < len(self.question_ids):  # 确保有三个问题
                    triplet = [self.question_ids[i], self.question_ids[i + 1], self.question_ids[i + 2]]
                    triplets.append(triplet)

            # 将三元组转换为张量（假设每个三元组的句子 ID 已经是准备好的输入）
            triplet_tensor = torch.stack([torch.stack(triplet) for triplet in triplets], dim=0)

            # 如果使用 GPU，迁移数据到 GPU
            if torch.cuda.is_available():
                triplet_tensor = triplet_tensor.cuda()

            # 通过模型计算所有三元组的向量
            self.knwb_vectors = self.model(triplet_tensor[:, 0], triplet_tensor[:, 1], triplet_tensor[:, 2])
            # 将所有向量进行归一化
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()

        # 假设 valid_data 返回的数据是 (sentence1, sentence2, sentence3, labels)
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况

            # 需要确保 input_id 是一个包含三个句子的三元组： (sentence1, sentence2, sentence3)
            if len(input_id) != 3:
                print(f"错误：batch_data 中的 input_id 不是三元组，而是： {input_id}")
                continue  # 跳过当前 batch

            # 解包三元组
            sentence1, sentence2, sentence3 = input_id

            # 使用模型进行预测（计算嵌入向量）
            with torch.no_grad():
                test_question_vectors = self.model(sentence1, sentence2, sentence3)

            # 计算预测结果并写入统计
            self.write_stats(test_question_vectors, labels)

        # 修改：防止除以零的错误
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        if correct + wrong > 0:
            accuracy = correct / (correct + wrong)
        else:
            accuracy = 0  # 防止分母为零

        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % accuracy)
        self.logger.info("--------------------")
        return

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
            if int(hit_index) == int(label):
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
        self.logger.info("--------------------")
        return
