
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"] 
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        # self.layer = nn.LSTM(hidden_size, hidden_size, num_layers=3, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = nn.functional.max_pool1d(x.transpose(1,2), x.shape[1]).squeeze()
        return x 
    
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = self.cosine_triplet_loss

    def cosine_distance(self, tensor1, tensor2):
        tensor1 = nn.functional.normalize(tensor1)
        tensor2 = nn.functional.normalize(tensor2)
        cosine = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
        return 1 - cosine 
    
    # 三元损失用于训练模型，在训练好SentenceEncoder之后，还是计算余弦相似度判断是否匹配
    def cosine_triplet_loss(self, a, p, n, margin=0.1):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        diff = ap - an + margin 
        return torch.mean(diff[diff.gt(0)])
    
    def forward(self, sentence1, sentence2=None, sentence3=None):
        if sentence2==None:
            return self.sentence_encoder(sentence1)
        sentence1 = self.sentence_encoder(sentence1)
        sentence2 = self.sentence_encoder(sentence2)
        if sentence3==None:
            return self.cosine_distance(sentence1, sentence2)
        sentence3 = self.sentence_encoder(sentence3)
        return self.loss(sentence1, sentence2, sentence3, margin=self.config["margin"])
    
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        return SGD(model.parameters(), lr=learning_rate)
    
if __name__=="__main__":
    from config import Config
    Config["vocab_size"] = 10
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [2,2,0,1]])
    s3 = torch.LongTensor([[2,4,1,0], [5,2,3,1]]) 
    loss = model(s1,s2,s3)
    print(loss)