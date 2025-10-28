# 利用RNN模型判断'我'字符在文本中的位置

''''
1. 构造词典，映射成embedding层
2. 模型训练

'''
import torch 
import torch.nn as nn
import json 
import random 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei"]  # 设置字体，解决中文显示问题
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class TorchModel(nn.Module):
    def __init__(self,num_embedding,embedding_dim,hidden_size,output_size):
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(num_embedding,embedding_dim,padding_idx=0)
        self.rnn = nn.RNN(embedding_dim,hidden_size,bias=True,batch_first=True)
        self.layer = nn.Linear(hidden_size,output_size)
        self.loss = nn.functional.cross_entropy
    
    def forward(self,x,y=None):
        # x:(batch_size,sentence_length) -> x:(batch_size,sentence_length,embedding_dim) , y:(batch_size,embedding_dim)
        x = self.embedding(x)
        output,hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        y_pred = self.layer(hidden)
        if y is not None:
            loss = self.loss(y_pred,y)
            return loss 
        return torch.softmax(y_pred,dim=-1)

def build_vocab():
    chars = '你我他abcdefg'
    vocab = {'pad':0}
    index = 0
    for char in chars:
        index += 1
        vocab[char] = index
    vocab['unk'] = index+1
    # 将词典导入json文件
    with open(r'vocab.json','w',encoding='utf-8') as f:
        json.dump(vocab,f,indent=2,ensure_ascii=False)
    return vocab

def sentence_to_seq(sentence,vocab,seq_length):
    seq = []
    sentence_length = len(sentence)
    if sentence_length>seq_length:
        sentence = sentence[:seq_length]
    for char in sentence:
        seq.append(vocab.get(char,vocab['unk']))
    if sentence_length <seq_length:
        seq += [vocab['pad']]*(seq_length-sentence_length)
    return seq

def build_sample(vocab,sentence_length,seq_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = x.index('我') if '我' in x else len(x)
    x = sentence_to_seq(x,vocab,seq_length)
    return x,y

def build_batch(batch_size,vocab,sentence_length,seq_length):
    X,Y = [],[]
    for i in range(batch_size):
        x,y = build_sample(vocab,sentence_length,seq_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X),torch.LongTensor(Y)

def evaluate(model,vocab,sentence_length,seq_length):
    x,y = build_batch(100,vocab,sentence_length,seq_length)
    correct,wrong = 0,0
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        for yp,yt in zip(y_pred,y):
            if np.argmax(yp) == yt:
                correct +=1
            else:
                wrong +=1
    return correct/(correct+wrong)

# 模型训练

def main():
    epoch_num = 20
    vocab = build_vocab()
    embedding_dim = 5
    sentence_length = 6
    seq_length = 6
    sample_num = 10000
    batch_size = 20
    model = TorchModel(num_embedding=len(vocab),embedding_dim=embedding_dim,hidden_size=128,output_size=seq_length+1)
    optim = torch.optim.Adam(model.parameters(),lr=0.001)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx in range(sample_num//batch_size):
            x,y = build_batch(batch_size,vocab,sentence_length,seq_length)
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(float(loss.item()))
        acc = evaluate(model,vocab,sentence_length,seq_length)
        print(f"第{epoch}轮，loss：{np.mean(watch_loss)}，acc：{acc}")
        log.append((acc,np.mean(watch_loss)))

    print(log[-1])
    # 绘图
    plt.title('训练过程中loss和精度的变化曲线')
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(),'rnn_model.bin')

if __name__ == '__main__':
    # main()
    with open(r'vocab.json',mode='r',encoding='utf-8') as f:
        vocab = json.load(f)
    input_strs = ['我sald','你阿斯兰噶的','saj我slgdj']
    input_vectors = []
    for str in input_strs:
        input_vectors.append(sentence_to_seq(str,vocab,seq_length=6))
    model = TorchModel(num_embedding=len(vocab),embedding_dim=5,hidden_size=128,output_size=7)
    model.load_state_dict(torch.load('rnn_model.bin'))
    y_pred = model(torch.LongTensor(input_vectors))
    for input_str,yp in zip(input_strs,y_pred):
        if torch.argmax(yp) == len(input_str):
            print(f"输入：{input_str}，预测结果：{yp}，预测下标：不存在‘我’")
        else:
            print(f"输入：{input_str}，预测结果：{yp}，预测下标：{torch.argmax(yp)}")



