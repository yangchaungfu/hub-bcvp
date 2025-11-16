import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现rnn网络完成一个简单nlp任务
判断文本中某些特定字符所在的索引，决定该向量为几类，都未匹配，则归为0，多个匹配，只取第一个索引

"""


class TorchModel(nn.Module):
    # vector_dim：每个字需要转成几维向量；sentence_length：样本文本长度；vocab：词表
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.pool = nn.RNN(sentence_length, len(vocab), bias=False, batch_first=True)   #RNN
        self.linear = nn.Linear(len(vocab), sentence_length)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)   # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)
        output, h = self.pool(x)   # (sen_len, batch_size) -> (1, batch_size)
        h_last = h[-1]
        x = self.linear(h_last)  # (batch_size, input_size) -> (batch_size, sentence_length)
        # print(x)
        # print(x.shape)
        x = x.squeeze(1)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x , dim=1)  # 使用softmax激活函数，实现多分类概率为1


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
    # print(vocab)
    # print("-----这是词表------")
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，不重复
    x = random.sample(list(vocab.keys()), k=min(sentence_length, len(vocab)))
    # print(x)
    # print("---随机样本----")
    #输入的字符出现时，返回其所在索引
    target_chars = {"你", "我", "他"}
    y = 0 # 默认值：无匹配时返回 0
    for idx, char in enumerate(x):
        if char in target_chars:
            y = idx  # 找到第一个匹配项，记录索引并退出循环
            break  # 若想返回所有匹配项的索引，可注释 break，用列表存储
    # all_indices = [idx for idx, char in enumerate(x) if char in target_chars]
    # 结果：有匹配则返回所有索引列表，无匹配则返回 [0]
    # y = all_indices if all_indices else [0]
    # print(y)
    # print(f"所有匹配字符的索引：{y}")  # 示例输出：[0, 2, 4]
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    # print(x,y)
    # print("-----这是样本------")
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
        # print(dataset_y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(100, vocab, sample_length)   #建立100个用于测试的样本
    print("100个样本中有索引为0的%d个，索引为1的%d个，索引为2的%d个，索引为3的%d个，索引为4的%d个，索引为5的%d个" % (sum(y == 0), sum(y == 1), sum(y == 2), sum(y == 3), sum(y == 4), sum(y == 5)))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 5  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 5  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        pred_class = torch.argmax(result[i]).item()  # 取概率最大的索引作为预测类别
        max_prob = torch.max(result[i]).item()  # 提取最大索引所在概率值
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, pred_class, max_prob))  # 打印结果


if __name__ == "__main__":
    main()
    # test_strings = ["fnvf我你", "wz你dfg", "rqwdet", "n我kwws", "dddwe他", "ike他hh"]
    # predict("model.pth", "vocab.json", test_strings)
