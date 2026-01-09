# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer

import loader
from config import Config
from model import TorchModel
import evaluate

"""
模型的测试、应用
"""


class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.tokenizer.vocab[" "] = len(self.tokenizer.vocab)  # 在词表中增加空格符
        self.tokenizer.vocab["[EOS]"] = len(self.tokenizer.vocab)  # 在词表中语句结束标识符

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()

    def predict(self, ask):
        """
        Args:
            ask: 问答系统的问句
        Returns:
            问答系统的答句
        """
        return evaluate.generate_answer(ask, 100, 30, self.tokenizer, model)


if __name__ == "__main__":
    model = TorchModel(Config)
    model.load_state_dict(torch.load("output/epoch_8.pth"))  # 加载模型参数

    pd = Predictor(Config, model)

    ask = "中宣部昨天召开“践行雷锋精神”新闻发布会，明确指出学习雷锋精神是当前加强社会思想道德建设的需要，今后的学雷锋以青少年为重点，以社会志愿服务为载体，通过办论坛、进教材、开微博等九大方面举措确保学雷锋常态化。（北京晨报）"
    ans = pd.predict(ask)
    print("ask >>", ask)
    print("ans <<", ans)

    while True:
        ask = input("请输入问句：")
        ans = pd.predict(ask)
        print("\t> " + ans)
