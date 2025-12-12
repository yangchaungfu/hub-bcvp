import os
import json
from collections import defaultdict

import jieba
import numpy as np
from bm25 import BM25
from similarity_function import editing_distance, jaccard_distance
from gensim.models import Word2Vec
import torch
from loader import load_data
from config import Config
from model import SiameseNetwork

'''
基于faq知识库和文本匹配算法进行意图识别，完成单轮问答
'''

class QASystem:
    def __init__(self, know_base_path, algo, config=None):
        '''
        :param know_base_path: 知识库文件路径
        :param algo: 选择不同的算法
        '''
        self.load_know_base(know_base_path)
        self.algo = algo
        if algo == "bm25":
            self.load_bm25()
        elif algo == "word2vec":
            self.load_word2vec()
        elif algo == "triplet":
            self.load_triplet_loss_model(know_base_path, config)
        else:
            #其余的算法不需要做事先计算
            pass

    def load_bm25(self):
        self.corpus = {}
        for target, questions in self.target_to_questions.items():
            self.corpus[target] = []
            for question in questions:
                self.corpus[target] += jieba.lcut(question)
        self.bm25_model = BM25(self.corpus)

    #词向量的训练
    def load_word2vec(self):
        #词向量的训练需要一定时间，如果之前训练过，我们就直接读取训练好的模型
        #注意如果数据集更换了，应当重新训练
        #当然，也可以收集一份大量的通用的语料，训练一个通用词向量模型。一般少量数据来训练效果不会太理想
        if os.path.isfile("model.w2v"):
            self.w2v_model = Word2Vec.load("model.w2v")
        else:
            #训练语料的准备，把所有问题分词后连在一起
            corpus = []
            for questions in self.target_to_questions.values():
                for question in questions:
                    corpus.append(jieba.lcut(question))
            #调用第三方库训练模型
            self.w2v_model = Word2Vec(corpus, vector_size=100, min_count=1) # 忽略总频率低于这个值的词，设置为1意味着所有词都会被考虑
            #保存模型
            self.w2v_model.save("model.w2v")
        #借助词向量模型，将知识库中的问题向量化
        self.target_to_vectors = {}
        for target, questions in self.target_to_questions.items():
            vectors = []
            for question in questions:
                vectors.append(self.sentence_to_vec(question))
            self.target_to_vectors[target] = np.array(vectors)

    # 将文本向量化
    def sentence_to_vec(self, sentence):
        vector = np.zeros(self.w2v_model.vector_size)
        words = jieba.lcut(sentence)
        # 所有词的向量相加求平均，作为句子向量
        count = 0
        for word in words:
            if word in self.w2v_model.wv:
                count += 1
                vector += self.w2v_model.wv[word]
        vector = np.array(vector) / count
        #文本向量做l2归一化，方便计算cos距离
        vector = vector / np.sqrt(np.sum(np.square(vector)))
        return vector

    def load_know_base(self, know_base_path):
        self.target_to_questions = {}
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.target_to_questions[target] = questions
        return

    def load_triplet_loss_model(self, know_base_path, config):
        self.config = config
        self.train_data = load_data(know_base_path, config)
        self.model = SiameseNetwork(config)
        self.model.load_state_dict(torch.load("model_output/epoch_20.pth"))
        self.knwb_to_vector()

    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            # if torch.cuda.is_available():
            #     question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            # 将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    def query(self, user_query):
        results = []
        if self.algo == "editing_distance":
            for target, questions in self.target_to_questions.items():
                scores = [editing_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "jaccard_distance":
            for target, questions in self.target_to_questions.items():
                scores = [jaccard_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "bm25":
            words = jieba.lcut(user_query)
            results = self.bm25_model.get_scores(words)
        elif self.algo == "word2vec":
            query_vector = self.sentence_to_vec(user_query)
            for target, vectors in self.target_to_vectors.items():
                cos = query_vector.dot(vectors.transpose())
                # print(cos)
                results.append([target, np.mean(cos)])
        elif self.algo == "triplet":
            input_id = self.encode_sentence(user_query)
            input_id = torch.LongTensor([input_id])
            # if torch.cuda.is_available():
            #     input_id = input_id.cuda()
            with torch.no_grad():
                query_vector = self.model(input_id)
                query_vector = torch.nn.functional.normalize(query_vector, dim=-1)  #  v / |v| 用户问题向量归一化
                res = torch.mm(query_vector.unsqueeze(0), self.knwb_vectors.T).squeeze()
                hit_index = int(torch.argmax(res))  # 命中问题标号
                hit_index = self.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
                standard_question = self.index_to_standard_question[hit_index]  # 对应的标准问题
                return standard_question
        else:
            assert "unknown algorithm!!"
        sort_results = sorted(results, key=lambda x:x[1], reverse=True)
        return sort_results[:3]


if __name__ == '__main__':
    qas = QASystem("data/train.json", "triplet", Config)
    question = "我的账户里面还有多少流量"
    # question = "我已经交足了话费请立即帮我开机"
    # question = "办理协议预存款活动有什么条件"
    res = qas.query(question)
    print(question)
    print(res)
    #
    # while True:
    #     question = input("请输入问题：")
    #     res = qas.query(question)
    #     print("命中问题：", res)
    #     print("-----------")

