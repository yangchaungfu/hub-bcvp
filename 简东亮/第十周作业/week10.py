#coding:utf8

导入 torch
导入 torch.nn 作为 nn
导入 numpy 库并将其命名为 np
导入数学
导入随机数
导入操作系统
导入 re
from  transformers  import  BertModel , BertTokenizer
"""
基于pytorch的LSTM语言模型
"""


class  LanguageModel ( nn.Module ) :​
    def  __init__ ( self , input_dim , vocab ):
        super ( LanguageModel , self ).__ init__ ()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.bert = BertModel.from_pretrained ( r " F:\AI\BaDou\bert-base- chinese  " ) 
        self.classifier = nn.Linear ( 768 , input_dim )​​ ​​ 
        self.classify = nn.Linear ( input_dim , len ( vocab ) )​​ ​ 
        self.dropout = nn.Dropout ( 0.1 )​​ ​​ 
        self.loss = nn.function.cross_entropy​​ ​​​​ 

    #当输入真实标签，返回损失值；无真实标签，返回预测值
    def  forward ( self , x , y = None , attention_mask = None ):
        # x = self.embedding(x) #输出形状：(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x) #输出形状:(batch_size, sen_len, input_dim)
        x , _  =  self.bert (​​
            input_ids=x，
            注意力掩码=注意力掩码
        ) #输出形状:(批次大小, 序列长度, 输入维度)
        打印（x .形状）
        x  =  self.classifier ( x )​​
        打印（x .形状）
        y_pred = self.classify(x) #输出形状：(batch_size, sen_len, vocab_size)
        如果 y 不为 None：
            return  self.loss ( y_pred.view ( -1 , y_pred.shape [ -1 ] ) , y.view ( -1 ) )​​​​​​​​
        别的：
            返回 torch.softmax(y_pred, dim=-1)

#加载字表
def  build_vocab ( vocab_path ):
    词汇表 = {"<pad>":0}
    with  open ( vocab_path , encoding = "utf8" ) as  f :
        对于索引，行在 enumerate(f) 中：
            char = line[:-1] #去掉结尾换行符
            vocab [ char ] =  index  +  1  #留出0位给pad token
    返回词汇表

#加载语料
def  load_corpus ( path ):
    语料库 = ""
    with  open ( path , encoding = "gbk" ) as  f :
        对于 f 中的每行：
            corpus  +  = line.strip ( )
    返回语料库

# 随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def  build_sample ( vocab , window_size , corpus ):
    start  =  random.randint ( 0 , len ( corpus ) - 1 - window_size )​​   
    结束 = 开始 + 窗口大小
    窗口 = 语料库[开始:结束]
    target = corpus[start + 1:end + 1] #输入输出错开一位
    # print(window, target)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window] #将字转换成序号
    x = [窗口中逐字逐字] #将字转换成序号
    y  = [ vocab . get ( word , vocab [ "<UNK>" ]) for  word  in  target ]
    返回 x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab词表
#window_size 样本长度
#corpus 语料字符串
def  build_dataset ( sample_length , vocab , window_size , corpus ):
    dataset_x  = []
    dataset_y  = []
    for  i  in  range ( sample_length ):
        x , y  =  build_sample ( vocab , window_size , corpus )
        dataset_x.append ( x )​​
        dataset_y.append ( y )​​
    # 初始化BERT分词器
    tokenizer  =  BertTokenizer.from_pretrained ( r"F:\AI\BaDou\bert-base- chinese " )

    # 使用tokenizer处理字符串
    编码 = 分词器(
        dataset_x，
        padding=True，
        截断=True，
        max_length=window_size, #限制长度
        add_special_tokens=False, # 不添加[CLS]/[SEP]
        return_tensors = "pt"
    ）

    # 获取输入ID和注意力掩码
    input_ids  =  encoded [ "input_ids" ]
    attention_mask  =  encoded [ "attention_mask" ]
    返回 input_ids、torch.LongTensor(dataset_y) 和 attention_mask

#建立模型
def  build_model ( vocab , char_dim ):
    model  =  LanguageModel ( char_dim , vocab )
    返回模型

#文本生成测试代码
def  generate_sentence ( openings , model , vocab , window_size ):
    reverse_vocab  =  dict (( y , x ) for x  , y in  vocab.items  ( ) )
    模型.评估()
    使用 torch.no_grad()：
        pred_char  =  ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while  pred_char  !=  " \n "  and  len ( openings ) <=  30 :
            openings  +=  pred_char
            x  = [ vocab.get ( char , vocab [ " <UNK>" ]) for char in openings [ -window_size : ] ]   
            x  =  torch.LongTensor ( [ x ] )
            如果 torch.cuda.is_available():
                x  =  x.cuda ( )​
            y  =  model ( x )[ 0 ][ - 1 ]
            index  =  sampling_strategy ( y )
            pred_char  =  reverse_vocab [ index ]
    返回开口

def  sampling_strategy ( prob_distribution ):
    如果 random.random() > 0.1：
        策略 = “贪婪”
    别的：
        策略 = “抽样”

    如果策略 == "贪婪"：
        返回 int(torch.argmax(prob_distribution))
    elif strategy == "采样":
        prob_distribution  =  prob_distribution.cpu ( ). numpy ( )
        返回 np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def  calc_perplexity ( sentence , model , vocab , window_size ):
    概率 = 0
    模型.评估()
    使用 torch.no_grad()：
        for  i  in  range ( 1 , len ( sentence )):
            起始位置 = max(0, i - 窗口大小)
            window  =  sentence [ start : i ]
            x  = [ vocab.get ( char , vocab [ " <UNK>" ] ) for char in window ]   
            x  =  torch.LongTensor ( [ x ] )
            目标 = sentence[i]
            target_index  =  vocab.get ( target , vocab [ "<UNK> " ] )
            如果 torch.cuda.is_available():
                x  =  x.cuda ( )​
            pred_prob_distribute  =  model ( x )[ 0 ][ - 1 ]
            target_prob  =  pred_prob_distribute [ target_index ]
            prob  + =  math.log ( target_prob , 10 )​
    返回 2 ** (prob * ( -1 / len(sentence)))


def  train ( corpus_path , save_weight = True ):
    #epoch_num = 20 #训练轮数
    epoch_num = 5 # 训练轮数
    batch_size = 64 #每次训练样本个数
    # train_sample = 50000 # 每轮训练总共训练的样本总数
    train_sample = 6400 # 每轮训练总共的样本总数
    char_dim = 256 #每个字的维度
    window_size = 10 # 样本文本长度
    vocab = build_vocab("vocab.txt") #建立字表
    corpus = load_corpus(corpus_path) #加载语料
    model = build_model(vocab, char_dim) #建立模型
    如果 torch.cuda.is_available():
        model  =  model.cuda ( )​
    optim = torch.optim.Adam(model.parameters(), lr=0.01) #建立优化器
    print ( "文本词表模型加载完毕，开始训练" )
    for  epoch  in  range ( epoch_num ):
        模型.训练()
        watch_loss  = []
        for  batch  in  range ( int ( train_sample  /  batch_size )):
            x, y, Attention_mask = build_dataset(batch_size, vocab, window_size, corpus) # 构建一组训练样本
            如果 torch.cuda.is_available():
                x、y、attention_mask = x.cuda()、y.cuda()、attention_mask.cuda()
            optim.zero_grad() #梯度归零
            loss = model(x, y, Attention_mask) # 计算loss
            loss.backward() #计算梯度
            optim.step() #更新权重
            watch_loss.append ( loss.item ( ) )​​
        print("==========\n第%d轮平均损失:%f" % (epoch + 1, np.mean(watch_loss)))
        print ( generate_sentence ( "让他在半年之前，就不能做出" , model , vocab , window_size ))
        print ( generate_sentence ( "李慕站在山路上，深深的呼吸" , model , vocab , window_size ))
    如果未保存重量：
        返回
    别的：
        base_name  =  os.path.basename ( corpus_path ) .replace ( " txt " , " pth " )
        model_path  =  os.path.join ( " model " , base_name )​​
        torch.save ( model.state_dict ( ) , model_path )​​
        返回



如果 __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train ( "corpus.txt" , False )
