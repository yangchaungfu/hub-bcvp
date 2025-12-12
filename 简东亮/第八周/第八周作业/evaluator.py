import torch 
from loader import load_data 

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config 
        self.model = model 
        self.logger = logger 
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct":0, "wrong":0}

    # 记录问题到标准问题的映射，用于确认匹配的是否正确。
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                standard_question_index = self.train_data.dataset.schema[standard_question]
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)

        # 将所有问题合并成矩阵，方便并行计算余弦距离
        with torch.no_grad():
            question_matrix = torch.stack(self.question_ids, dim=0)
            self.knwb_vectors = self.model(question_matrix)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return 

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            input_id, labels = batch_data 
            with torch.no_grad():
                test_question_vectors = self.model(input_id)
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return 

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # torch.mm只能计算二维矩阵相乘
            try:
                res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            except Exception as e:
                print(f"torch.mm有误：{test_question_vector.shape} {self.knwb_vectors.T.shape}")
                raise(e)
            hit_index = int(torch.argmax(res.squeeze()))   # 得到命中问题编号
            hit_index = self.question_index_to_standard_question_index[hit_index]
            if int(hit_index) == int(label):
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
        return       
