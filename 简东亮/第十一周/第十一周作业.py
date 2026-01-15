# coding:utf8

导入 torch
import  torch.nn as nn ​​ 
导入 numpy 库 并将其命名为 np
导入 数学
导入 随机数
导入 操作系统
导入 re
from  transformers  import  BertModel , BertConfig , BertTokenizer

"""
基于pytorch的Bert语言模型
生成任务
"""


class  LanguageModel ( nn.Module ) :​
    def  __init__ ( self , vocab ):
        super ( LanguageModel , self ).__ init__ ()
        model_path  =  r"D:\bert-base-chinese"
        self.bert_config = BertConfig.from_pretrained ( model_path )​​ ​​ 
        自己。伯特_配置。num_hidden_​​layers  =  6   # 设置 Bert 层数
        自己。Bert  =  BertModel ( config = self . bert_config )   # 实例化模型
        self.classify = nn.Linear ( self.bert_config.hidden_ ​​size , len ( vocab ) )​ ​​​​​ 
        self.dropout = nn.Dropout ( 0.1 )​​ ​​ 
        self.loss = nn.function.cross_entropy​​ ​​​​ 

    # 当输入真实标签，返回损失值；无真实标签，返回预测值
    def  forward ( self , x , attention_mask = None , y = None ):
        # mask = torch.LongTensor(torch.tril(torch.ones(x.shape[0], x.shape[1], x.shape[1])))
        # mask_final = torch.matmul(attention_mask, mask)
        output  =  self.Bert ( x , attention_mask = atteaching_mask )   #输出形状：(batch_size, sen_len, hidden_​​size )
        x_last  = 自我. dropout ( output .last_hidden_ ​​state )   # 随机神经元，按比例放大其余神经元
        y_pred  =  self.classify ( x_last )   #输出形状：(batch_size, sen_len, vocab_size )
        如果 y 不 为 None：
            return  self.loss ( y_pred.view ( -1 , y_pred.shape [ -1 ] ) , y.view ( -1 ) , ignore_index = -100 )​​​​​​​​​
        别的：
            返回 torch.softmax ( y_pred , dim = -1 )​​​


# 加载字表
def  build_vocab ( vocab_path ):
    词汇表 = { "<pad>" : 0 }
    with  open ( vocab_path , encoding = "utf8" ) as  f :
        对于 索引，行在 enumerate  ( f ) :
            char  =  line [: - 1 ]   # 去掉结尾换行符
            vocab [ char ] =  index  +  1   # 留出0位给pad token
    返回 词汇表


# 加载语料
def  load_corpus ( path ):   # 将语料去除空格输出
    语料库 =  ""
    with  open ( path , encoding = "GBK" ) as  f :
        对于 f中的每行 ： 
            corpus  +  = line.strip ( )
    返回 语料库


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def  build_sample ( vocab , window_size , corpus , tokenizer ):
    开始 = 随机。randint ( 0 , len ( corpus ) -  1  -  window_size )   # 随机窗口启动位置
    end  =  start  +  window_size   # 窗口结束位置
    window  =  corpus [ start : end ]   # 确定窗口中的实际字符串
    target  =  corpus [ start  +  1 : end  +  1 ]   # 输入错输出开一位

    # 输入编码和目标
    x  =  tokenizer.encode_plus ( window ,​​
                              max_length = window_size  +  1 ,
                              padding = "max_length" ,
                              截断= True，
                              return_attention_mask = True ,
                              add_special_tokens = False )

    y  =  tokenizer.encode_plus ( target ,​​
                              max_length = window_size  +  1 ,
                              padding = "max_length" ,
                              截断= True，
                              return_attention_mask = True ,
                              add_special_tokens = False )

    x_input_ids  =  x [ "input_ids" ]
    x_attention_mask  =  x [ "attention_mask" ]
    y_input_ids  =  y [ "input_ids" ]

    #设置标签，只有答案部分有效
    labels  = [ - 100 ] * ( window_size  //  2 ) +  y_input_ids [ window_size  //  2 :]

    返回 x_input_ids、x_attention_mask和labels


# 建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
# 词汇词表
# window_size 样本长度
# 语料库 语料字符串
def  build_dataset ( sample_length , vocab , window_size , corpus , tokenizer ):
    dataset_x_input_ids  = []
    dataset_x_attention_mask  = []
    dataset_y_labels  = []
    for  i  in  range ( sample_length ):
        x_input_id , x_att_mask , y_label \
            =  build_sample ( vocab , window_size , corpus , tokenizer )
        dataset_x_input_ids.append ( x_input_id )​​
        dataset_x_attention_mask。附加（x_att_mask）
        dataset_y_labels.append ( y_label )​​
    返回（
        torch.LongTensor ( dataset_x_input_ids ) ,​
        torch.LongTensor ( dataset_x_attention_mask ) ,​
        torch.LongTensor ( dataset_y_labels )​​
    ）


# 建立模型
def  build_model ( vocab ):
    model  =  LanguageModel ( vocab )
    返回 模型


# 文本生成测试代码
def  generate_sentence ( openings , model , tokenizer , window_size ):
    模型.评估()
    使用 torch.no_grad ( ) ：
        pred_char  =  ""
        while  pred_char  !=  " \n "  and  len ( openings ) <=  30 :
            openings  +=  pred_char
            x  =  tokenizer.encode_plus ( openings ,​​
                                      max_length = window_size，
                                      padding = False ,
                                      截断= True，
                                      return_attention_mask = True ,
                                      add_special_tokens = False )
            x_input_ids  =  x [ "input_ids" ]
            x_in  =  torch.LongTensor ( [ x_input_ids ] )
            如果 torch.cuda.is_available ( ) :​​
                x_in  =  x_in.cuda ( )​
            # print(x_in.shape)
            y  =  model ( x_in )[ 0 ][ - 1 ]   # 提取映射的字表维度的计算
            index  =  Sample_Strategy ( y )   # 利用贪心算法得出概率最大的那个字
            pred_char  =  tokenizer.decode ( [ index ]) [ 0 ]
    返回 开口


def  sampling_strategy ( prob_distribution ):
    如果 random.random ( ) >  0.1：​
        策略 =  “贪婪”
    别的：
        策略 =  “抽样”

    如果 策略 ==  "贪婪"：
        返回 int ( torch.argmax ( prob_distribution ) )​
    elif  strategy  ==  "sampling" :
        prob_distribution  =  prob_distribution.cpu ( ). numpy ( )
        返回 np.random.choice ( list ( range ( len ( prob_distribution ) ) ) , p = prob_distribution )​


# 计算文本ppl
def  calc_perplexity ( sentence , model , tokenizer , window_size ):
    概率 =  0
    模型.评估()
    使用 torch.no_grad ( ) ：
        for  i  in  range ( 1 , len ( sentence )):
            start  =  max ( 0 , i  -  window_size )
            window  =  sentence [ start : i ]
            x  =  tokenizer.encode_plus ( window ,​​
                                      max_length = window_size，
                                      padding = "max_length" ,
                                      截断= True，
                                      return_attention_mask = True ,
                                      add_special_tokens = False )
            x_input_ids  =  x [ "input_ids" ]
            x_in  =  torch.LongTensor ( [ x_input_ids ] )
            目标 = 句子[ i ]
            target_index  =  tokenizer.convert_tokens_to_ids ( target )​​
            如果 torch.cuda.is_available ( ) :​​
                x_in  =  x_in.cuda ( )​
            pred_prob_distribute  =  model ( x_in )[ 0 ][ - 1 ]
            target_prob  =  pred_prob_distribute [ target_index ]
            prob  + =  math.log ( target_prob , 10 )​
    返回 2  ** ( prob  * ( - 1  /  len ( sentence )))


def  train ( corpus_path , save_weight = False ):
    epoch_num  =  20   # 训练轮数
    batch_size  =  32   # 批量训练样本个数
    train_sample  =  10000   # 每轮训练总共训练的样本总数
    window_size  =  32   # 样本文本长度
    vocab  =  build_vocab ( "vocab_bert.txt" )   # 建立字表
    corpus  =  load_corpus ( corpus_path )   # 加载语料
    tokenizer  = ( BertTokenizer.from_pretrained )​
                 （r"D:\bert-base-chinese" , do_lower_case = True）
    model  =  build_model ( vocab )   # 建立模型
    如果 torch.cuda.is_available ( ) :​​
        model  =  model.cuda ( )​
    优化 = 火炬.优化。Adam ( model .parameters (), lr = 2e-5 )   # 建立优化器
    print ( "文本词表模型加载完毕，开始训练" )
    for  epoch  in  range ( epoch_num ):
        模型.训练()
        watch_loss  = []
        for  batch  in  range ( int ( train_sample  /  batch_size )):
            x , x_att , y  =  build_dataset ( batch_size , vocab , window_size , corpus , tokenizer )   # 构建一组训练样本
            如果 torch.cuda.is_available ( ) :​​
                x , x_att , y  =  x。CUDA（），x_att。CUDA (), y . CUDA（）
            优化。Zero_grad ()   # 梯度归零
            loss  =  model ( x , x_att , y )   # 计算loss
            损失。back ()   # 计算梯度
            优化。step ()   #更新权重
            watch_loss.append ( loss.item ( ) )​​
        print ( "========== \n第%d轮平均损失:%f"  % ( epoch +  1  , np .mean ( watch_loss ) ))
        print ( generate_sentence ( "李清脚步声，似乎是明白了" , model , tokenizer , window_size ))
        print ( generate_sentence ( "李慕站在山路上，深深的呼吸" , model , tokenizer , window_size ))
    如果 save_weight：
        model_path  = 操作系统.小路。join ( r"E:\AI_NLP课程\第十一周大模型相关第一讲\sft\models" ,
                                  "DIY_nnlm_bert.pth" )
        torch.save ( model.state_dict ( ) , model_path )​​
        print ( f"模型权重已保存至{ model_path } " )
        返回
    别的：
        返回


如果 __name__  ==  "__main__" :
    train ( "corpus.txt" , True )
页脚
