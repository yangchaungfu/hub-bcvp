#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
在nlpdemo中使用rnn模型训练，判断特定字符在文本中的位置
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层

        self.layer = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)

        # 分类层：将RNN输出映射到类别数（0-5,6,7共8类）
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()  # 多分类用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # 2. RNN: 取最后一个时间步的隐藏状态作为全局特征
        _, hidden = self.layer(x)  # hidden形状: (1, batch_size, hidden_size)
        hidden = hidden.squeeze(0)  # 压缩为(batch_size, hidden_size)
        # 3. 分类: (batch_size, hidden_size) -> (batch_size, num_classes)
        y_pred = self.classifier(hidden)
        # print("==================",y_pred)
        # print("------------------",y)
        if y is not None:
            return self.loss(y_pred, y.long())   #预测值和真实值计算损失 标签转为long类型
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "a我cdefghijklmnopqrstuvwxyz_"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    print(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 统计"我"字出现的次数和位置
    me_indices = [i for i, word in enumerate(x) if word == "我"]
    count = len(me_indices)
    if count == 1:
        y = me_indices[0]  # 只出现一次，y为其索引
    elif count > 1:
        y = sentence_length + 1  # 出现多次，y = sentence_length + 1
    else:
        y = sentence_length  # 未出现，y= sentence_length
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
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length, hidden_size):
    num_classes = sentence_length + 2  # 计算类别数量（0~5 + 6 +7 = 8类）
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size, num_classes)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        pred_classes = torch.argmax(y_pred, dim=1)  # 取最大概率类别
        for y_p, y_t in zip(pred_classes, y):  #与真实标签进行对比

            if y_p == y_t:
                correct += 1  # 预测正确
            else:
                wrong += 1  # 预测错误
    # print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    acc = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{acc:.4f}")
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    hidden_size = 16
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_size,)
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
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_size = 16
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length, hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        # 统一输入长度（不足补pad，过长截断）
        if len(input_string) < sentence_length:
            input_string += "_" * (sentence_length - len(input_string))
        else:
            input_string = input_string[:sentence_length]
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        y_pred = model(torch.LongTensor(x))
        pred_classes = torch.argmax(y_pred, dim=1)
        pred_probs = torch.softmax(y_pred, dim=1).max(dim=1).values  # 最大概率
    for i, input_string in enumerate(input_strings):
        pred = pred_classes[i].item()
        prob = pred_probs[i].item()
        # 解析预测结果
        if pred < sentence_length:
            desc = f"位置{pred}"
        elif pred == sentence_length:
            desc = "未出现"
        else:
            desc = "多次出现"
        print("输入：%s, 预测类别：%d, %s, 概率值：%f" % (input_string, pred, desc, prob)) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww","erqwe我","af我我塔塔","io我","我sdkfjsdffwew"]
    predict("model.pth", "vocab.json", test_strings)
