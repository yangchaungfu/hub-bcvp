#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的RNN网络编写
实现一个网络完成判断特定字符在文本中的位置任务
文本中有6个位置，若是包含你/我/他，则在第几个索引位置就属于第几类(1-6)，若没有则属于第0类
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.sentence_length = sentence_length
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim * sentence_length, num_classes)  # 线性层
        self.activation = nn.Softmax(dim=-1)  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, hidden = self.rnn(x)  # (batch_size, sen_len, vector_dim)
        # 展平rnn输出以适配全连接层
        rnn_out = rnn_out.reshape(rnn_out.shape[0], -1)  # (batch_size, sen_len*vector_dim)
        output = self.classify(rnn_out)  # (batch_size, num_classes)
        y_pred = self.activation(output)  # (batch_size, num_classes)
        
        if y is not None:
            return self.loss(output, y)
        else:
            return y_pred  # 输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) 
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#目标：判断"你我他"字符在句子中的位置
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    
    #标签：0表示没有目标字符，1-6表示"你我他"在对应位置(1-based)
    y = 0
    target_chars = {"你", "我", "他"}
    
    #查找第一个目标字符的位置
    for i, char in enumerate(x):
        if char in target_chars:
            y = i + 1  # 位置从1开始计数
            break
            
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
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
def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchModel(char_dim, sentence_length, vocab, num_classes)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length, sentence_length, num_classes):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)   #建立200个用于测试的样本
    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测 (batch_size, num_classes)
        y_pred = torch.argmax(y_pred, dim=-1)  #获取最大概率的类别 (batch_size)
        correct += (y_pred == y).sum().item()
        total += y.numel()  # 总元素数
    acc = correct / total
    print("正确预测个数：%d, 总数：%d, 正确率：%f"%(correct, total, acc))
    return acc

def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 1000   #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    num_classes = 7       #分类数：0(无目标字符), 1-6(目标字符位置)
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, sentence_length, num_classes)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return model, vocab

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20         # 每个字的维度
    sentence_length = 6   # 样本文本长度
    num_classes = 7       # 分类数
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length, num_classes)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        # 确保字符串长度为sentence_length
        if len(input_string) < sentence_length:
            # 使用空格进行填充
            input_string = input_string.ljust(sentence_length)
        elif len(input_string) > sentence_length:
            input_string = input_string[:sentence_length]
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
    
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        result = torch.argmax(result, dim=-1)  # 获取每个样本最可能的类别
        
    for i, input_string in enumerate(input_strings):
        pred_class = result[i].item()
        if pred_class == 0:
            print("输入：%s, 预测结果：目标属于第0类" % input_string)
        else:
            print("输入：%s, 预测结果：目标属于第%d类" % (input_string, pred_class))

if __name__ == "__main__":
    # 训练模型
    model, vocab = main()
    
    # 测试预测
    test_strings = ["fnvf我e", "wz你dfg", "他rqwde", "n我k你他", "abcdef"]
    predict("rnn_model.pth", "vocab.json", test_strings)