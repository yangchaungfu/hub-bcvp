#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
11维判断：x是一个10维向量，判断文本中特定字符出现的位置，“你”出现在一句话的第几个位置下标，就是第几类（11分类任务，如果特定字符未出现，就是第10类）
输出：概率分布  11维向量【0.03,0.07,0.1,0.05,0.05,0.28,0.14,0.07,0.02,0.03,0.16】，每个下标的元素表示第几类（0~9）的概率，如果未出现，就是第10类
"""

class TorchModel(nn.Module):
    def __init__(self, vocab, sentence_length, vector_dim, hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)   #RNN层
        self.classify = nn.Linear(hidden_size, 11)     #线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失 nn.CrossEntropyLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, h = self.rnn(x)                    #(batch_size, sen_len, vector_dim) -> (1, batch_size, hidden_size)
        # 根据PyTorch官网，RNN的输入输出在batch_first=True时，(N, L, H_in) -> RNN -> (D * num_layers, N, H_out)
        # N = batch_size, L = sen_len, H_in = input_size, H_out = hidden_size，D = 2 if bidirectional=True otherwise 1，
        # num_layers = Number of recurrent layers. Default: 1

        # x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)

        x = h.squeeze()                            #(1, batch_size, hidden_size) -> (batch_size, hidden_size)
        y_pred = self.classify(x)                  #(batch_size, hidden_size) -> (batch_size, 11)
        # y_pred = self.activation(x)              #(batch_size, 11) -> (batch_size, 11)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, axis=-1)   #输出预测结果-概率分布

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
        # Process the sequence through RNN layer
    chars = "中国欢迎你fghijklmnopqrstuvwxyz金木水火土"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) 
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # print(f"随机x生成样本：{"".join(x)}")
    label = "你"
    # label位于哪个下标，就是哪一类（第0~seq_len-1类）
    if label in x:
        y = x.index(label)  # list.index() 方法只会返回第一个匹配元素的下标，即使列表中有多个相同的元素。如果需要获取所有匹配元素的下标，
    # 指定字未出现，则为第seq_len类
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    # print(f"x : {x}, y : {y}")
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, sentence_length, char_dim, hidden_size):
    model = TorchModel(vocab, sentence_length, char_dim, hidden_size)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if np.argmax(y_p) == int(y_t):
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 800    #每轮训练总共训练的样本总数
    char_dim = 16         #每个字的维度
    hidden_size = 32      #RNN隐藏层的维度
    sentence_length = 10  #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, sentence_length, char_dim, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本 (batch_size, sen_len) => (batch_size, 1)
            # print(f"本批次X、Y样本：{x} --> {y}")  # torch.Size([20, 6]) --> torch.Size([20, 1])
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            # optim.zero_grad()    #梯度归零（这一步放在最后这里也可以）
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss) / batch_size)])

    #保存模型
    torch.save(model.state_dict(), "model2.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc", color='green')  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss", color='orange')  # 画loss曲线
    plt.legend()
    plt.show()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 16    # 每个字的维度
    sentence_length = 10  # 样本文本长度
    hidden_size = 32     # 隐藏层大小
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, sentence_length, char_dim, hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
        # print(f"输入字符串 --> 词表 --> 序列化: {input_string} --> {x[-1]}")
        model.eval()   #测试模式
        with torch.no_grad():  #不计算梯度
            result = model.forward(torch.LongTensor(x))  #模型预测
        # print(result)
    for str, res in zip(test_strings, result):
        formatted_data = [f"{x:.6f}" for x in res]
        print("输入：%s, 预测类别：%s,\n概率值：%s" % (str, torch.argmax(res).numpy(), formatted_data))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fn1国87v我e2", "wz你gtdfyut", "rsqqi9mmx5", "ju@你&5中)vp", "国欢ttonxmy你",
                    "国fwlx国中p国w", "你pyu国pv国zm", "qivw你u&wd3", "kzsql国pkt你", "欢ialfp迎osw",
                    "h8fvz欢你ruq", "on^@m%中你*m", "uz#$%^&zjz", "n99$迎f!k迎r", "llsm你你j你rk",
                    "中t国rgip你迎u", "中你sm国你ghin", "w国mxis迎w你k", "ijm国q35中it", "v30k6你90h+"]
    print("=====================预测结果======================")
    predict("model2.pth", "vocab.json", test_strings)
