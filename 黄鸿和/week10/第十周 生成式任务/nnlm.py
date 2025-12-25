import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel

#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    """
    计算文本的困惑度（Perplexity, PPL）
    PPL是语言模型的评估指标，值越低表示模型对文本的预测越准确
    Args:
        sentence (str): 待评估的文本
        model (LanguageModel): 训练好的语言模型
        vocab (dict): 字表字典
        window_size (int): 输入窗口大小（与训练一致）
    Returns:
        float: 文本的困惑度值
    计算公式：
        PPL = 2^(-1/N * Σ(log2 P(w_i | w_1...w_{i-1})))
        其中N为文本长度，P(w_i)为模型预测第i个字符的概率
    说明：
        - 对每个字符，用其前面的字符（最多window_size个）预测当前字符
        - 取log10后转换为log2计算（因最终结果用2为底）
    """
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))



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

def load_corpus(corpus_path):
	corpus = ""
	with open(corpus_path, encoding = "gbk") as f:
		for line in f:
			corpus += line.strip()
	return corpus

def build_model(vocab, char_dim, encoder_layers):
	model = LanguageModel(char_dim, vocab, encoder_layers)
	return model

def build_sample(vocab, window_size, corpus):
	start_index = random.randint(0, len(corpus) -1 - window_size)
	end_index = start_index + window_size
	sampling_window = corpus[start_index:end_index]
	target = corpus[start_index + 1 : end_index + 1] #目标文本比输入文本向后移动一个字

	x = [vocab.get(char, vocab["<UNK>"]) for char in sampling_window]
	y = [vocab.get(char, vocab["<UNK>"]) for char in target]
	return x, y


def build_dataset(sample_length, vocab, window_size, corpus):
	dataset_x = []
	dataset_y = []
	for i in range(sample_length):
		x, y = build_sample(vocab, window_size, corpus)
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


def generate_sentence(openings_text, model, vocab, window_size):
	# 解码，idx到字的映射
	ix_to_char = {ix:char for char, ix in vocab.items()}
	model.eval()
	with torch.no_grad():
		# 一个个字进行预测
		pred_char = ""
		# 生成了换行符，或生成文本超过30字则终止迭代
		while pred_char != "\n" and len(openings_text) <=30:
			openings_text += pred_char
			x = [vocab.get(char, vocab["<UNK>"]) for char in openings_text[-window_size:]]
			x = torch.LongTensor([x])
			if torch.cuda.is_available():
				x = x.cuda()
			y = model(x)[0][-1]
			index = sampling_strategy(y)
			pred_char = ix_to_char[index]
	return openings_text





def train(corpus_path, save_weight = True):
	epoch_num = 50        #训练轮数
	batch_size = 64       #每次训练样本个数
	train_sample = 50000   #每轮训练总共训练的样本总数
	char_dim = 768        #每个字的维度
	window_size = 10       #样本文本长度
	encoder_layers = 3
	vocab = build_vocab("vocab.txt")       #建立字表
	corpus = load_corpus(corpus_path)  #加载语料
	model = build_model(vocab, char_dim, encoder_layers)    #建立模型
	if torch.cuda.is_available():
		model = model.cuda()
	# 学习率变小
	optim = torch.optim.Adam(model.parameters(), lr = 0.001)  #优化器
	print("文本词表模型加载完成，开始训练...")
	for epoch in range(epoch_num):
		model.train()
		watch_loss = []
		for batch in range(int(train_sample / batch_size)):
			x, y = build_dataset(batch_size, vocab, window_size, corpus)
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

		print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
		print(generate_sentence("李慕站在山路上，深深的呼吸，", model, vocab, window_size))
	if not save_weight:
		return 
	else:
		torch.save(model.state_dict(), "nnlm_model.pth")
		print("模型权重保存完成！")


if __name__ == "__main__":
	train("corpus.txt", False)