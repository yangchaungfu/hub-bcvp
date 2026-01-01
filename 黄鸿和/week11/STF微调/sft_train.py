import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel



class LanguageModel(nn.Module):
	def __init__(self, input_dim = 768, vocab = None, use_bert_layers = 1):
		"""
        Args:
            input_dim: BERT隐藏层维度（固定768）
            vocab: 字表字典
            use_bert_layers: 使用BERT的前N层（1≤N≤12）
        """
		super(LanguageModel, self).__init__()
		# 加载完整BERT预训练模型
		full_bert = BertModel.from_pretrained(r"F:\八斗学院\bert-base-chinese")
		self.bert_embeddings = full_bert.embeddings
		self.bert_encoder_layers = nn.Sequential(*full_bert.encoder.layer[:use_bert_layers])

		self.classify = nn.Linear(input_dim, len(vocab))
		self.dropout = nn.Dropout(0.1)
		self.loss = nn.functional.cross_entropy
	# 生成 mask 掩码
	def generate_causal_mask(self, seq_len, device):
		"""
        生成因果掩码（下三角矩阵）
        :param seq_len: 序列长度
        :param device: 设备
        :return: [1, seq_len, seq_len]的掩码矩阵
    """
		mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
		return mask.unsqueeze(0)  # 增加batch维度
	
	def forward(self, x, y = None):
		batch_size, seq_len = x.shape
		device = x.device
		embeddings_output = self.bert_embeddings(x)
		# 2. 生成因果掩码（阻止看到未来token）
		causal_mask = self.generate_causal_mask(seq_len, device)  # [1, seq_len, seq_len]

		encoder_output = embeddings_output
		# 取出每层BERT编码器进行前向计算
		for layer_module in self.bert_encoder_layers:
			encoder_output = layer_module(
				hidden_states = encoder_output,
				attention_mask = causal_mask
				)[0]

		encoder_output = self.dropout(encoder_output)
		y_pred = self.classify(encoder_output)

		if y is not None:
			return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
		else:
			return torch.softmax(y_pred, dim = -1)

def build_vocab(vocab_path):
	vocab = {"<pad>" : 0} #特殊填充符号
	with open(vocab_path, encoding="utf-8") as f:
		for index, line in enumerate(f):
			char = line[:-1]  #去掉结尾换行符
			vocab[char] = index + 1
	return vocab


def build_model(vocab, char_dim, encoder_layers):
	model = LanguageModel(char_dim, vocab, encoder_layers)
	return model

def build_stf_sample(vocab, corpus, max_input_len, max_output_len):
	item = random.choice(corpus)
	input_text = item["instruction"].strip()
	output_text = item["output"].strip()

	# print('input_text', input_text)
	# print('output_text', output_text)


	x = [vocab.get(char, vocab["<UNK>"]) for char in input_text[:max_input_len]]
	x += [vocab["<pad>"]] * (max_input_len - len(x))
	y = [vocab.get(char, vocab["<UNK>"]) for char in output_text[:max_output_len]]
	y += [vocab["<pad>"]] * (max_output_len - len(y))
	return x, y


def build_sft_dataset(sample_length, vocab, corpus, max_input_len, max_output_len):
	dataset_x = []
	dataset_y = []
	for i in range(sample_length):
		x, y = build_stf_sample(vocab, corpus, max_input_len, max_output_len)
		dataset_x.append(x)
		dataset_y.append(y)
	return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def sampling_strategy(y_prob_distribution):
	if random.random() > 0.1:
		strategy = "greedy"
	else:
		strategy = "sampling"
	
	if strategy == "greedy":
		return int(torch.argmax(y_prob_distribution))
	elif strategy == "sampling":
		y_prob_distribution = y_prob_distribution.cpu().numpy()
		sampling_text = np.random.choice(list(range(len(y_prob_distribution))), p = y_prob_distribution)
		return sampling_text


def generate_sentence(openings_text, model, vocab):
	# 解码，idx到字的映射
	ix_to_char = {ix:char for char, ix in vocab.items()}
	model.eval()
	with torch.no_grad():
		# 一个个字进行预测
		pred_char = ""
		# 生成了换行符，或生成文本超过30字则终止迭代
		while pred_char != "\n" and len(openings_text) <=30:
			openings_text += pred_char
			x = [vocab.get(char, vocab["<UNK>"]) for char in openings_text]
			x = torch.LongTensor([x])
			if torch.cuda.is_available():
				x = x.cuda()
			y = model(x)[0][-1]
			index = sampling_strategy(y)
			pred_char = ix_to_char[index]
	return openings_text





def train(save_weight = True):
	corpus = [
        {
            "instruction": "请介绍一下人工智能。",
            "input": "",
            "output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、感知和自然语言处理等能力。"
        },
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "深度学习是机器学习的一个子领域，使用具有多个层（深度）的神经网络来学习数据的复杂模式和表示。它模仿人脑的神经网络结构。"
        },
        {
            "instruction": "Python中的列表和元组有什么区别？",
            "input": "",
            "output": "列表（list）是可变的，可以修改、添加或删除元素，使用方括号[]。元组（tuple）是不可变的，一旦创建就不能修改，使用圆括号()。"
        },
        {
            "instruction": "解释一下什么是监督学习。",
            "input": "",
            "output": "监督学习是机器学习的一种方法，使用标记的训练数据来训练模型。模型学习输入和输出之间的映射关系，然后可以对新的未标记数据进行预测。"
        },
        {
            "instruction": "如何提高模型的泛化能力？",
            "input": "",
            "output": "提高模型泛化能力的方法包括：1) 增加训练数据量和多样性 2) 使用正则化技术（如Dropout、L2正则化）3) 数据增强 4) 交叉验证 5) 防止过拟合。"
        },
        {
            "instruction": "什么是Transformer模型？",
            "input": "",
            "output": "Transformer是一种基于注意力机制的神经网络架构，由编码器和解码器组成。它摒弃了RNN和CNN，完全依赖注意力机制来处理序列数据，成为现代NLP的基础架构。"
        },
        {
            "instruction": "请解释一下梯度下降算法。",
            "input": "",
            "output": "梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数对参数的梯度，然后沿着梯度反方向更新参数，逐步接近最优解。学习率控制每次更新的步长。"
        },
        {
            "instruction": "什么是迁移学习？",
            "input": "",
            "output": "迁移学习是将在一个任务或领域上学到的知识应用到另一个相关任务上的技术。它允许模型利用预训练的知识，从而在目标任务上更快地学习和获得更好的性能。"
        },
        {
            "instruction": "如何处理自然语言处理中的文本分类问题？",
            "input": "",
            "output": "文本分类的常见步骤包括：1) 文本预处理（分词、去停用词）2) 特征提取（词袋、TF-IDF、词向量）3) 选择分类算法（朴素贝叶斯、SVM、神经网络）4) 训练和评估模型。"
        },
        {
            "instruction": "请介绍一下大语言模型。",
            "input": "",
            "output": "大语言模型（LLM）是拥有数十亿甚至千亿参数的深度学习模型，通过在海量文本数据上预训练获得语言理解能力。它们可以执行各种NLP任务，如文本生成、问答、翻译等。"
        }
    ]
	epoch_num = 10        #训练轮数
	batch_size = 64       #每次训练样本个数
	train_sample = 50000   #每轮训练总共训练的样本总数
	char_dim = 768        #每个字的维度
	encoder_layers = 10
	max_input_len = 50
	max_output_len = 50
	vocab = build_vocab(r"F:\八斗学院\作业\黄鸿和-week11\vocab.txt")       #建立字表
	model = build_model(vocab, char_dim, encoder_layers)    #建立模型
	if torch.cuda.is_available():
		model = model.cuda()
	# 学习率变小
	optim = torch.optim.Adam(model.parameters(), lr = 0.001)  #优化器
	print("模型加载完成，开始训练...")
	print("================================================")
	print(generate_sentence("请介绍一下人工智能。", model, vocab))
	print(generate_sentence("什么是Agent。", model, vocab))
	print("================================================")
	print("开始SFT训练...")
	for epoch in range(epoch_num):
		model.train()
		watch_loss = []
		for batch in range(int(train_sample / batch_size)):
			x, y = build_sft_dataset(batch_size, vocab, corpus, max_input_len, max_output_len)
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			optim.zero_grad()
			loss = model(x, y)
			loss.backward()
			optim.step()
			watch_loss.append(loss.item())
		print("============第{}轮训练完成，平均损失：{:.4f}============".format(epoch + 1, np.mean(watch_loss)))
		print("===============================================")

		print(generate_sentence("请介绍一下人工智能。", model, vocab))
		print(generate_sentence("什么是Agent", model, vocab))
	if not save_weight:
		return 
	else:
		torch.save(model.state_dict(), "sft_bert_model.pth")
		print("模型权重保存完成！")


if __name__ == "__main__":
	train(True)  #设置为True保存模型权重
	# train(False)  #设置为False不保存模型权重
