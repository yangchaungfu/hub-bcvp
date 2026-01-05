# -*- coding: utf-8 -*-

导入 torch
导入 torch.nn 作为 nn
from  torch.optim import  Adam , SGD​​ 
from  torchcrf  import  CRF
from  transformers  import  BertModel
"""
建立网络模型结构
"""

class  TorchModel ( nn.Module ) :​
    def  __init__ ( self , config ):
        超级（TorchModel，自我）。__init__ ()
        hidden_​​size = config["hidden_​​size"]
        vocab_size  =  config [ "vocab_size" ] +  1
        max_length  =  config [ "max_length" ]
        class_num  =  config [ "class_num" ]
        层数 = 配置["层数"]
        # self.embedding = nn.Embedding(vocab_size, hidden_​​size, padding_idx=0)
        self.bert_like = BertModel.from_pretrained  ( config [ " bert_path " ] , return_dict = False )​ 
        hidden_​​size = self.bert_like.config.hidden_​​size
        # self.layer = nn.LSTM(hidden_​​size, hidden_​​size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(hidden_​​size, class_num)
        self.crf_layer = CRF ( class_num , batch_first = True )​​  
        self.use_crf = config [ " use_crf " ]  
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #loss采用交叉熵损失

    #当输入真实标签，返回损失值；无真实标签，返回预测值
    def  forward ( self , x , target = None ):
        # x = self.embedding(x) #输入形状：(batch_size, sen_len)
        # x, _ = self.layer(x) #输入形状:(batch_size, sen_len, hidden_​​size * 2)
        x, _ = self.bert_like(x) # (batch_size, sen_len) -> (batch_size, sen_len, hidden_​​size), (batch_size, hidden_​​size)
        predict = self.classify(x) # 输出：(batch_size, sen_len, hidden_​​size) -> (batch_size * sen_len, num_tags)

        如果 target 不为 None：
            如果 self.use_crf：
                mask  =  target.gt ( -1 )​​​
                # pytorch-crf 库返回的计算结果不是取相反数的S(X, y) - log(Z(X))，这里需要取反才是loss = log(Z(X)) - S(X, y)
                返回 - self.crf_layer(predict, target, mask, reduction="mean")
            别的：
                #(数字, 类号), (数字)
                # 交叉熵对输入有形状要求(number, class_num), (number)，需要查看变换张量形状
                # (batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
                返回 self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        别的：
            如果 self.use_crf：
                return self.crf_layer.decode(predict) # 解码出来是一个序列(batch_size, sen_len) （在评估的时候钢铁包标签序列还原为文本段）
            别的：
                返回预测


def  choose_optimizer ( config , model ):
    优化器 = 配置["优化器"]
    learning_rate  =  config [ "learning_rate" ]
    如果优化器 == "adam":
        返回 Adam(model.parameters(), lr=learning_rate)
    elif  optimizer  ==  "sgd" :
        返回 SGD(model.parameters(), lr=learning_rate)


如果 __name__ == "__main__":
    from  config  import  Config
    配置["vocab_size"] = 20
    model  =  TorchModel ( Config )
