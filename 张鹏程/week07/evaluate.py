import torch
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger, data_type="valid", device="cpu"):
        """
        初始化评估器
        
        Args:
            config: 配置字典
            model: 模型实例
            logger: 日志记录器
            data_type: 数据类型，"valid"表示验证集，"test"表示测试集
            device: 设备类型，如"cuda"、"mps"、"cpu"
        """
        self.config = config
        self.model = model
        self.logger = logger
        self.data_type = data_type
        self.device = device
        
        # 根据数据类型选择数据路径
        if data_type == "test":
            data_path = config.get("test_data_path", config["valid_data_path"])
            self.data_type_name = "测试"
        else:
            data_path = config["valid_data_path"]
            self.data_type_name = "验证"
        
        self.valid_data = load_data(data_path, config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info(f"开始{self.data_type_name}第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            # 将数据迁移到对应设备
            batch_data = [d.to(self.device) for d in batch_data]
            # 处理BERT和非BERT模型的不同输入格式
            if self.config["model_type"] in ["bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]:
                input_ids, attention_mask, labels = batch_data
                with torch.no_grad():
                    pred_results = self.model(input_ids, attention_mask=attention_mask)
            else:
                input_ids, labels = batch_data
                with torch.no_grad():
                    pred_results = self.model(input_ids)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

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
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
