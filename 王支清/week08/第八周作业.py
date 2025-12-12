import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.optim import Adam, SGD

# 配置参数信息
Config = {
    "model_path": "model_output",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 3,  # 减少epoch数以便快速测试
    "batch_size": 16,
    "epoch_data_size": 100,  # 每轮训练中采样数量
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "margin": 0.1,
    "vocab_size": 100,  # 词汇表大小
}


class DataGenerator(Dataset):
    """数据生成器 - 完全使用模拟数据"""

    def __init__(self, config, data_type="train"):
        self.config = config
        self.vocab = self.create_vocab()
        self.schema = self.create_schema()
        self.config["vocab_size"] = len(self.vocab)
        self.data_type = data_type
        self.data = []
        self.knwb = defaultdict(list)
        self.load()

    def create_vocab(self):
        """创建词汇表"""
        vocab = {}
        # 创建一些示例字符
        sample_chars = ["[PAD]", "[UNK]", "你", "好", "吗", "我", "爱", "学", "习", "机",
                        "器", "深", "度", "是", "什", "么", "人", "工", "智", "能"]

        for index, char in enumerate(sample_chars):
            vocab[char] = index

        # 添加一些数字
        for i in range(10):
            vocab[str(i)] = len(vocab)

        # 添加一些英文字母
        for i in range(26):
            vocab[chr(ord('a') + i)] = len(vocab)
            vocab[chr(ord('A') + i)] = len(vocab)

        return vocab

    def create_schema(self):
        """创建schema（类别映射）"""
        schema = {
            "问候": 0,
            "学习": 1,
            "技术": 2,
            "其他": 3
        }
        return schema

    def create_sample_sentences(self):
        """创建示例句子"""
        # 不同类别的示例句子
        category_sentences = {
            "问候": ["你好", "你好吗", "最近怎么样", "早上好", "晚上好", "嗨", "哈喽"],
            "学习": ["如何学习", "学习的方法", "怎样学习", "学习技巧", "高效学习"],
            "技术": ["Python编程", "TensorFlow使用", "PyTorch教程", "模型训练"],
            "其他": ["天气怎么样", "今天星期几", "你的名字", "谢谢", "再见"]
        }
        return category_sentences

    def load(self):
        """加载数据 - 生成模拟数据"""
        category_sentences = self.create_sample_sentences()

        if self.data_type == "train":
            # 训练数据：构建知识库
            for category, sentences in category_sentences.items():
                for sentence in sentences:
                    input_id = self.encode_sentence(sentence)
                    input_id = torch.LongTensor(input_id)
                    self.knwb[category].append(input_id)
        else:
            # 验证/测试数据
            for category, sentences in category_sentences.items():
                for sentence in sentences:
                    input_id = self.encode_sentence(sentence)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[category]])
                    self.data.append([input_id, label_index])

            # 如果数据太少，复制一些
            while len(self.data) < self.config["epoch_data_size"] // 2:
                self.data.extend(self.data)

    def encode_sentence(self, sentence):
        """将句子编码为ID序列"""
        input_id = []
        for char in sentence:
            # 使用词汇表，如果不存在则使用UNK
            input_id.append(self.vocab.get(char, self.vocab.get("[UNK]", 1)))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        """填充序列到固定长度"""
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            return self.data[index % len(self.data)]

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            return max(len(self.data), self.config["epoch_data_size"] // 2)

    def random_train_sample(self):
        """随机采样训练样本（三元组）"""
        categories = list(self.knwb.keys())

        # 确保至少有两个类别
        if len(categories) < 2:
            categories = categories * 2

        # 随机选择两个不同的类别
        category_a, category_b = random.sample(categories, 2)

        # 锚点：使用类别名称作为锚点
        anchor_text = category_a
        anchor = self.encode_sentence(anchor_text)
        anchor = torch.LongTensor(anchor)

        # 正样本：从同一类别中随机选择
        if len(self.knwb[category_a]) == 0:
            # 如果没有正样本，创建一个简单的
            positive = torch.zeros(self.config["max_length"], dtype=torch.long)
        else:
            positive = random.choice(self.knwb[category_a])

        # 负样本：从不同类别中随机选择
        if len(self.knwb[category_b]) == 0:
            # 如果没有负样本，创建一个
            negative = torch.ones(self.config["max_length"], dtype=torch.long)
        else:
            negative = random.choice(self.knwb[category_b])

        return [anchor, positive, negative]


def load_data(config, shuffle=True, data_type="train"):
    """加载数据"""
    dg = DataGenerator(config, data_type=data_type)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


class SentenceEncoder(nn.Module):
    """句子编码器"""

    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        x = self.dropout(x)
        x = self.layer1(x)  # (batch_size, seq_len, hidden_size)

        # 最大池化
        if x.dim() == 3:
            x = F.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(-1)
        return x


class SiameseNetwork(nn.Module):
    """孪生网络"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sentence_encoder = SentenceEncoder(config)

    def cosine_distance(self, tensor1, tensor2):
        """计算余弦距离"""
        # 归一化
        tensor1 = F.normalize(tensor1, dim=-1)
        tensor2 = F.normalize(tensor2, dim=-1)
        # 计算余弦相似度
        cosine = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
        # 余弦距离 = 1 - 余弦相似度
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=0.1):
        """三元组损失"""
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        diff = ap - an + margin
        # 只考虑diff > 0的部分
        loss = torch.mean(torch.clamp(diff, min=0))
        return loss

    def forward(self, sentence1, sentence2=None, sentence3=None):
        """前向传播"""
        if sentence2 is None:
            return self.sentence_encoder(sentence1)

        sentence1_encoded = self.sentence_encoder(sentence1)
        sentence2_encoded = self.sentence_encoder(sentence2)

        if sentence3 is None:
            return self.cosine_distance(sentence1_encoded, sentence2_encoded)

        sentence3_encoded = self.sentence_encoder(sentence3)
        return self.cosine_triplet_loss(sentence1_encoded, sentence2_encoded,
                                        sentence3_encoded, margin=self.config["margin"])


def choose_optimizer(config, model):
    """选择优化器"""
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        return SGD(model.parameters(), lr=learning_rate)


class Evaluator:
    """评估器"""

    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config, shuffle=False, data_type="valid")
        self.train_data = load_data(config, data_type="train")
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.knwb_vectors = None
        self.question_index_to_standard_question_index = None

    def knwb_to_vector(self):
        """将知识库中的问题转换为向量"""
        self.question_index_to_standard_question_index = {}
        question_ids = []

        # 获取训练数据生成器
        train_generator = self.train_data.dataset

        # 遍历知识库
        for idx, (category, question_tensors) in enumerate(train_generator.knwb.items()):
            for question_tensor in question_tensors:
                standard_question_index = train_generator.schema[category]
                self.question_index_to_standard_question_index[len(question_ids)] = standard_question_index
                question_ids.append(question_tensor)

        # 将所有问题合并成矩阵，方便并行计算余弦距离
        if question_ids:
            with torch.no_grad():
                question_matrix = torch.stack(question_ids, dim=0)
                self.knwb_vectors = self.model(question_matrix)
                self.knwb_vectors = F.normalize(self.knwb_vectors, dim=-1)
        else:
            # 如果没有问题，创建一个虚拟的
            self.knwb_vectors = torch.randn(10, self.config["hidden_size"])
            self.knwb_vectors = F.normalize(self.knwb_vectors, dim=-1)
            for i in range(10):
                self.question_index_to_standard_question_index[i] = i % 4

    def eval(self, epoch):
        """评估模型"""
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.model.eval()
        self.knwb_to_vector()

        total_samples = 0
        for index, batch_data in enumerate(self.valid_data):
            if index >= 3:  # 只评估前几个batch
                break

            input_id, labels = batch_data
            with torch.no_grad():
                test_question_vectors = self.model(input_id)
            self.write_stats(test_question_vectors, labels)
            total_samples += len(labels)

        self.show_stats()
        return

    def write_stats(self, test_question_vectors, labels):
        """记录统计结果"""
        assert len(labels) == len(test_question_vectors)

        for test_question_vector, label in zip(test_question_vectors, labels):
            if self.knwb_vectors is None:
                self.stats_dict["wrong"] += 1
                continue

            try:
                # 计算与知识库中所有向量的相似度
                similarity = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
                hit_index = int(torch.argmax(similarity.squeeze()))

                # 获取命中的标准问题索引
                if hit_index in self.question_index_to_standard_question_index:
                    hit_standard_index = self.question_index_to_standard_question_index[hit_index]

                    # 检查是否匹配正确
                    if int(hit_standard_index) == int(label.item()):
                        self.stats_dict["correct"] += 1
                    else:
                        self.stats_dict["wrong"] += 1
                else:
                    self.stats_dict["wrong"] += 1

            except Exception as e:
                self.logger.error(f"计算相似度时出错: {e}")
                self.stats_dict["wrong"] += 1

        return

    def show_stats(self):
        """显示统计结果"""
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong

        if total > 0:
            self.logger.info("预测集合条目总量：%d" % total)
            self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
            self.logger.info("预测准确率：%f" % (correct / total))
        else:
            self.logger.info("没有进行预测")

        self.logger.info("--------------------")
        return


def main(config):
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载数据
    logger.info("加载训练数据...")
    train_data = load_data(config, data_type="train")

    # 初始化模型和优化器
    logger.info("初始化模型...")
    model = SiameseNetwork(config)
    optimizer = choose_optimizer(config, model)

    # 初始化评估器
    logger.info("初始化评估器...")
    evaluator = Evaluator(config, model, logger)

    # 训练循环
    logger.info("开始训练...")
    for epoch in range(config["epoch"]):
        epoch_num = epoch + 1
        model.train()
        logger.info("epoch %d 开始" % epoch_num)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if index >= 10:  # 每个epoch只训练前10个batch，加快速度
                break

            optimizer.zero_grad()
            a, p, n = batch_data
            loss = model(a, p, n)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            if index % 5 == 0:
                logger.info(f"batch {index}, loss: {loss.item():.4f}")

        avg_loss = np.mean(train_loss) if train_loss else 0
        logger.info("epoch %d 平均损失: %f" % (epoch_num, avg_loss))

        # 评估模型
        evaluator.eval(epoch_num)

        # 保存模型
        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch_num)
        torch.save(model.state_dict(), model_path)
        logger.info("模型已保存到: %s" % model_path)

    logger.info("训练完成!")
    return


# 使用示例
if __name__ == "__main__":
    # 运行主函数
    main(Config)

    # 模型测试示例
    print("\n" + "=" * 50)
    print("模型测试示例:")
    print("=" * 50)

    # 测试模型
    model = SiameseNetwork(Config)

    # 创建测试数据
    test_sentences = [
        "你好",
        "如何学习",
        "Python编程",
        "天气怎么样"
    ]

    # 创建DataGenerator实例来编码句子
    dg = DataGenerator(Config, data_type="train")

    print("\n编码测试:")
    for sentence in test_sentences:
        encoded = dg.encode_sentence(sentence)
        print(f"句子: '{sentence}' -> 编码: {encoded[:10]}...")

    print("\n模型前向传播测试:")

    # 测试单句编码
    test_input = torch.LongTensor([dg.encode_sentence("你好"), dg.encode_sentence("学习")])
    encoded_output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"编码输出形状: {encoded_output.shape}")

    # 测试三元组损失
    anchor = torch.LongTensor([dg.encode_sentence("你好")])
    positive = torch.LongTensor([dg.encode_sentence("你好吗")])
    negative = torch.LongTensor([dg.encode_sentence("学习")])

    loss = model(anchor, positive, negative)
    print(f"\n三元组损失: {loss.item():.4f}")

    # 测试相似度计算
    s1 = torch.LongTensor([dg.encode_sentence("你好")])
    s2 = torch.LongTensor([dg.encode_sentence("你好吗")])
    s3 = torch.LongTensor([dg.encode_sentence("学习")])

    sim_12 = model(s1, s2)
    sim_13 = model(s1, s3)

    print(f"\n相似度测试:")
    print(f"'你好' 和 '你好吗' 的余弦距离: {sim_12.item():.4f}")
    print(f"'你好' 和 '学习' 的余弦距离: {sim_13.item():.4f}")
    print(f"相似度差异: {sim_13.item() - sim_12.item():.4f} (应该为正数，表示不相似)")
