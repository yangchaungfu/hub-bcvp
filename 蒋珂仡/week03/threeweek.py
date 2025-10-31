#coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json

class TorchModel(nn.Module):
    def __init__(self,vector_dim,vocab,hidden_size,num_classes):
        super(TorchModel,self).__init__()
        self.embedding=nn.Embedding(len(vocab),vector_dim,padding_idx=0)
        self.rnn=nn.RNN(input_size=vector_dim,hidden_size=hidden_size,batch_first=True)
        self.classify=nn.Linear(hidden_size,num_classes)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        x=self.embedding(x)
        run_output,hidden=self.rnn(x)
        hidden=hidden.squeeze(0)
        y_pred=self.classify(hidden)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab
def build_sample(vocab,sentence_length):
    target_char="你"
    x_chars=[random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y=sentence_length
    for i,char in enumerate(x_chars):
        if char==target_char:
            y=i
            break
    x=[vocab.get(word,vocab["unk"]) for word in x_chars]
    return x,y

def build_dataset(sample_length,vocab,sentence_length):
    dataset_x=[]
    dataset_y=[]
    for i in range(sample_length):
        x,y=build_sample(vocab,sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)


def build_model(voacb,char_dim,hidden_size,num_classes):
    model=TorchModel(char_dim,voacb,hidden_size,num_classes)
    return model

def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本

    # 统计正负样本（正样本：'你'出现；负样本：'你'未出现）
    pos_count = (y < sentence_length).sum().item()
    neg_count = (y == sentence_length).sum().item()
    print("本次预测集中共有%d个正样本('你'出现)，%d个负样本('你'未出现)" % (pos_count, neg_count))

    correct = 0
    with torch.no_grad():
        y_pred_logits = model(x)  # 模型预测，得到 logits
        # 8. 使用 argmax 找到概率最大的那个类别的索引
        y_pred = torch.argmax(y_pred_logits, dim=1)

        # 9. 对比预测的类别索引和真实的类别索引
        correct = (y_pred == y).sum().item()

    total = len(y)
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / total))
    return correct / total


def main():
    # 配置参数
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005

    # RNN 新增参数
    hidden_size = 32  # RNN 隐藏层维度
    # 分类任务新增参数
    num_classes = sentence_length + 1  # 类别数 = 6个位置 + 1个'未出现'

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, hidden_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
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
    char_dim = 20
    sentence_length = 6  # 必须和训练时一致

    # RNN 新增参数
    hidden_size = 32
    num_classes = sentence_length + 1

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 10. 序列化输入时，要确保长度统一为 sentence_length
    x = []
    for input_string in input_strings:
        # 截断或填充到 sentence_length
        input_string = input_string[:sentence_length]  # 截断
        input_string_padded = input_string + "pad" * (sentence_length - len(input_string))  # 填充

        x.append([vocab.get(char, vocab["unk"]) for char in input_string_padded])

    model.eval()
    with torch.no_grad():
        result_logits = model.forward(torch.LongTensor(x))
        # 11. 同样使用 argmax 获取预测类别
        result_classes = torch.argmax(result_logits, dim=1)

    for i, input_string in enumerate(input_strings):
        pred_class = result_classes[i].item()

        # 12. 解释预测的类别
        if pred_class == sentence_length:
            print("输入：%s, 预测: 未找到'你'" % (input_string))
        else:
            print("输入：%s, 预测: '你'在位置 %d" % (input_string, pred_class))


if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww", "你abcde"]
    predict("model.pth", "vocab.json", test_strings)




