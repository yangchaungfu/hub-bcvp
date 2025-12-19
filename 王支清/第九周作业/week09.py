# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import json
import re
import os
import random
import jieba
import logging
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from transformers import BertModel, BertTokenizer
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
配置 = {
    # 修改为相对路径或简单路径
    "model_path": "./model_output",
    "schema_path": "./ner_data/schema.json",
    "train_data_path": "./ner_data/train.txt",
    "valid_data_path": "./ner_data/test.txt",
    "vocab_path": "./bert-base-chinese/vocab.txt",  # 或使用本地vocab.txt
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 5,  # 测试时先用少量epoch
    "batch_size": 4,  # 测试时用小batch
    "优化器": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,  # 先设为False，等安装pytorch-crf后再改
    "class_num": 9,
    "bert_path": "bert-base-chinese",  # 可以是本地路径或模型名称
}

# ==================== 备用方案：如果缺少pytorch-crf ====================
try:
    from torchcrf import CRF

    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("警告：未安装pytorch-crf，将使用softmax代替")
    print("安装命令：pip install pytorch-crf")


# ==================== 数据加载模块 ====================
def load_vocab(vocab_path):
    """加载字表或词表"""
    token_dict = {}
    try:
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0 留给padding位置，所以从1开始
    except FileNotFoundError:
        print(f"警告：词汇表文件 {vocab_path} 未找到，使用BERT默认词表")
        tokenizer = BertTokenizer.from_pretrained(配置["bert_path"])
        token_dict = tokenizer.get_vocab()

    return token_dict


class DataGenerator(Dataset):
    """数据生成器"""

    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.data = []

        # 如果schema未找到，使用默认schema
        if not self.schema:
            self.schema = {
                "B-LOCATION": 0, "B-ORGANIZATION": 1, "B-PERSON": 2, "B-TIME": 3,
                "I-LOCATION": 4, "I-ORGANIZATION": 5, "I-PERSON": 6, "I-TIME": 7,
                "O": 8
            }

        self.load()

    def load(self):
        """加载数据"""
        try:
            with open(self.path, encoding="utf8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"警告：数据文件 {self.path} 未找到，创建虚拟数据")
            self.create_dummy_data()
            return

        segments = content.split("\n\n")
        for segment in segments:
            if not segment.strip():
                continue

            sentence = []
            labels = []
            for line in segment.split("\n"):
                if line.strip() == "":
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    char, label = parts[0], parts[1]
                    sentence.append(char)
                    labels.append(self.schema.get(label, 8))  # 默认O标签

            if sentence:
                self.sentences.append("".join(sentence))
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])

    def create_dummy_data(self):
        """创建虚拟数据用于测试"""
        dummy_sentences = [
            "张三在北京工作",
            "李四昨天去了上海",
            "王五在阿里巴巴公司上班",
            "今天是2023年10月1日"
        ]

        for sentence in dummy_sentences:
            self.sentences.append(sentence)
            input_ids = self.encode_sentence(list(sentence))
            # 创建虚拟标签
            labels = [8] * len(sentence)  # 全为O标签
            labels = self.padding(labels, -1)
            self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])

        print(f"创建了 {len(self.data)} 条虚拟数据")

    def encode_sentence(self, text, padding=True):
        """编码句子"""
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab.get("[UNK]", 100)))

        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id, pad_token=0):
        """补齐或截断输入的序列"""
        max_len = self.config["max_length"]
        if len(input_id) > max_len:
            input_id = input_id[:max_len]
        else:
            input_id = input_id + [pad_token] * (max_len - len(input_id))
        return input_id

    def load_schema(self, path):
        """加载标签schema"""
        try:
            with open(path, encoding="utf8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告：schema文件 {path} 未找到")
            return {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, shuffle=True):
    """用torch自带的DataLoader类封装数据"""
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


# ==================== 模型定义 ====================
class TorchModel(nn.Module):
    """序列标注模型（BERT + LSTM + CRF可选）"""

    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        class_num = config["class_num"]

        # 使用BERT作为编码器
        try:
            self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
            hidden_size = self.bert.config.hidden_size
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            print("使用随机初始化的Embedding代替")
            vocab_size = config["vocab_size"] + 1
            hidden_size = config["hidden_size"]
            self.bert = None
            self.embedding = nn.Embedding(vocab_size, hidden_size)

        # LSTM层
        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=config["num_layers"],
            dropout=0.1 if config["num_layers"] > 1 else 0
        )

        # 分类层
        self.classify = nn.Linear(hidden_size, class_num)

        # CRF层（如果可用）
        self.use_crf = config["use_crf"] and CRF_AVAILABLE
        if self.use_crf:
            self.crf_layer = CRF(class_num, batch_first=True)

        # 损失函数
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        """前向传播"""
        # BERT编码或Embedding
        if self.bert is not None:
            x, _ = self.bert(x)
        else:
            x = self.embedding(x)

        # LSTM层
        x, _ = self.lstm(x)

        # 分类预测
        predict = self.classify(x)  # (batch_size, sen_len, class_num)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return torch.argmax(predict, dim=-1)


def choose_optimizer(config, model):
    """选择优化器"""
    optimizer = config["优化器"]
    learning_rate = config["learning_rate"]

    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


# ==================== 评估模块 ====================
class Evaluator:
    """模型效果测试"""

    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }

    def eval(self, epoch):
        """评估模型"""
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()

        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]:(index + 1) * self.config["batch_size"]
                        ]

            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            input_id, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_id)

            self.write_stats(labels, pred_results, sentences)

        self.show_stats()

    def write_stats(self, labels, pred_results, sentences):
        """统计评估指标"""
        assert len(labels) == len(pred_results) == len(sentences)

        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if isinstance(pred_label, torch.Tensor):
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()

            # 只取有效长度
            valid_len = min(len(sentence), self.config["max_length"])
            pred_label = pred_label[:valid_len]
            true_label = true_label[:valid_len]

            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)

            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]]
                )
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])

    def show_stats(self):
        """显示评估结果"""
        if sum([self.stats_dict[key]["样本实体数"] for key in self.stats_dict]) == 0:
            self.logger.info("没有实体可评估，可能是数据问题")
            return

        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)

            self.logger.info(
                "%s类实体，精确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1)
            )

        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))

        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in self.stats_dict])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in self.stats_dict])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in self.stats_dict])

        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)

        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")

    def decode(self, sentence, labels):
        """解码标签序列"""
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)

        patterns = {
            "LOCATION": r"(04+)",
            "ORGANIZATION": r"(15+)",
            "PERSON": r"(26+)",
            "TIME": r"(37+)"
        }

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, labels):
                s, e = match.span()
                if s < len(sentence):
                    e = min(e, len(sentence))
                    results[entity_type].append(sentence[s:e])

        return results


# ==================== 训练主程序 ====================
def main(config):
    """模型训练主程序"""
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.makedirs(config["model_path"], exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("开始训练序列标注模型")
    logger.info(f"配置参数: {config}")

    # 加载训练数据
    logger.info("加载训练数据...")
    train_data = load_data(config["train_data_path"], config)
    logger.info(f"训练数据加载完成，共 {len(train_data.dataset)} 条样本")

    # 加载模型
    logger.info("初始化模型...")
    model = TorchModel(config)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可用，迁移模型至GPU")
        model = model.cuda()
    else:
        logger.info("使用CPU训练")

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    best_loss = float('inf')
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"Epoch {epoch}/{config['epoch']} 开始训练")

        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            input_id, labels = batch_data
            loss = model(input_id, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss.append(loss.item())

            if (index + 1) % 10 == 0:
                logger.info(f"批次 {index + 1}/{len(train_data)}, 损失: {loss.item():.4f}")

        avg_loss = np.mean(train_loss)
        logger.info(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(config["model_path"], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"保存最佳模型到 {model_path}")

        # 评估
        evaluator.eval(epoch)

    logger.info("训练完成!")
    return model, train_data


# ==================== 测试函数 ====================
def test_model():
    """测试模型"""
    print("\n=== 模型测试 ===")

    # 创建测试配置
    test_config = 配置.copy()
    test_config["batch_size"] = 2
    test_config["epoch"] = 1

    # 创建虚拟数据文件
    if not os.path.exists("./ner_data"):
        os.makedirs("./ner_data", exist_ok=True)

    # 创建虚拟schema
    schema = {
        "B-LOCATION": 0, "B-ORGANIZATION": 1, "B-PERSON": 2, "B-TIME": 3,
        "I-LOCATION": 4, "I-ORGANIZATION": 5, "I-PERSON": 6, "I-TIME": 7,
        "O": 8
    }

    with open("./ner_data/schema.json", "w", encoding="utf8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    # 创建虚拟训练数据
    train_data_content = """张 O
三 B-PERSON
在 O
北 B-LOCATION
京 I-LOCATION
工 O
作 O

李 O
四 B-PERSON
昨 B-TIME
天 I-TIME
去 O
了 O
上 B-LOCATION
海 I-LOCATION"""

    with open("./ner_data/train.txt", "w", encoding="utf8") as f:
        f.write(train_data_content)

    # 创建虚拟测试数据
    test_data_content = """王 O
五 B-PERSON
在 O
阿 B-ORGANIZATION
里 I-ORGANIZATION
巴 I-ORGANIZATION
巴 I-ORGANIZATION
公 O
司 O
上 O
班 O"""

    with open("./ner_data/test.txt", "w", encoding="utf8") as f:
        f.write(test_data_content)

    print("虚拟数据创建完成")
    print(f"训练数据: {len(train_data_content.split('\\n\\n'))} 条")
    print(f"测试数据: {len(test_data_content.split('\\n\\n'))} 条")

    # 运行训练
    model, _ = main(test_config)

    # 测试预测
    print("\n=== 测试预测 ===")
    model.eval()

    # 测试句子
    test_sentence = "张三在北京工作"
    tokenizer = BertTokenizer.from_pretrained(test_config["bert_path"])

    # 编码
    inputs = tokenizer.encode_plus(
        test_sentence,
        add_special_tokens=True,
        max_length=test_config["max_length"],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']

    with torch.no_grad():
        predictions = model(input_ids)

    print(f"测试句子: {test_sentence}")
    print(f"预测结果: {predictions[0]}")

    return model


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("序列标注模型 - 开始运行")
    print("=" * 50)

    # 检查依赖
    print("检查依赖...")
    try:
        import torch

        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        print("错误: 未安装PyTorch")
        print("安装命令: pip install torch")
        exit(1)

    try:
        from transformers import BertModel

        print("Transformers库: 已安装")
    except ImportError:
        print("错误: 未安装transformers")
        print("安装命令: pip install transformers")
        exit(1)

    # 创建必要目录
    os.makedirs("./model_output", exist_ok=True)
    os.makedirs("./ner_data", exist_ok=True)

    # 运行测试
    if not os.path.exists(配置["train_data_path"]):
        print("\n未找到训练数据文件，运行测试模式...")
        model = test_model()
    else:
        print("\n找到训练数据文件，开始正式训练...")
        model, train_data = main(配置)

    print("\n" + "=" * 50)
    print("程序运行完成!")
