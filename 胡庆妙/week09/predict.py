# -*- coding: utf-8 -*-
import re
import torch
import jieba
from config import Config
from model import TorchModel
from loader import load_vocab

"""
模型的测试、应用
"""


class Predictor:
    def __init__(self, config, vocab, model):
        self.config = config
        self.vocab = vocab
        self.model = model
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()

    def predict(self, sentence):
        input_ids = self.encode_sentence(sentence, padding=False)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        with torch.no_grad():
            pred_labels = self.model(torch.LongTensor([input_ids]))  # [1, sen_len] -> [1, sen_len]
        pred_labels = pred_labels[0].tolist()  # [sen_len]

        # 在本文中标记预测出来的实体，输出标记后的文本
        return self.label_entities_in_sentence(sentence, pred_labels)

    def encode_sentence(self, text, padding=True):
        input_ids = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_ids.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_ids = self.padding(input_ids)
        return input_ids

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["sentence_len"]]
        input_id += [pad_token] * (self.config["sentence_len"] - len(input_id))
        return input_id

    # 在本文中标注预测出来的实体，输出标注后的文本
    @staticmethod
    def label_entities_in_sentence(sentence, labels):
        """
        Args:
            sentence: 如： "他是彭德怀"
            labels: 如：[8, 8, 2, 6, 6]
        Returns:
            标注后的本文: 如：他是{彭德怀/PERSON}
        """
        labels = "".join([str(x) for x in labels])
        print("\tlabels: ", labels)
        labeled_sentence = ""
        idx = 0
        for segment in re.finditer("(04+)", labels):
            s, e = segment.span()  # 匹配到的子字符串的起始和结束位置
            labeled_sentence += sentence[idx:s] + "{" + sentence[s:e] + "/LOCATION}"  # 在原文中添加标记
            idx = e
        labeled_sentence += sentence[idx:]
        sentence = labeled_sentence

        labeled_sentence = ""
        idx = 0
        for segment in re.finditer("(15+)", labels):
            s, e = segment.span()
            labeled_sentence += sentence[idx:s] + "{" + sentence[s:e] + "/ORGANIZATION}"
            idx = e
        labeled_sentence += sentence[idx:]
        sentence = labeled_sentence

        labeled_sentence = ""
        idx = 0
        for segment in re.finditer("(26+)", labels):
            s, e = segment.span()
            labeled_sentence += sentence[idx:s] + "{" + sentence[s:e] + "/PERSON}"
            idx = e
        labeled_sentence += sentence[idx:]
        sentence = labeled_sentence

        labeled_sentence = ""
        idx = 0
        for segment in re.finditer("(37+)", labels):
            s, e = segment.span()
            labeled_sentence += sentence[idx:s] + "{" + sentence[s:e] + "/TIME}"
            idx = e
        labeled_sentence += sentence[idx:]

        return labeled_sentence


if __name__ == "__main__":
    vocab = load_vocab(Config["vocab_path"])  # 加载词表

    model = TorchModel(Config)  # 加载模型
    model.load_state_dict(torch.load("output/epoch_10.pth"))

    pd = Predictor(Config, vocab, model)
    input_text = ("建设海南自由贸易港的战略目标，就是要把海南自由贸易港打造成为引领我国新时代对外开放的重要门户。"
                "2025年11月6日，习近平总书记在海南省三亚市听取海南自由贸易港建设工作汇报并发表重要讲话。")
    print(">>", input_text)
    res = pd.predict(input_text)
    print("<<", res)
