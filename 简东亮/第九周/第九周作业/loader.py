# -*- coding: utf-8 -*-

导入 json
导入 re
导入 操作系统
导入 torch
导入 随机数
进口 杰巴
导入 numpy 库 并将其命名为 np
from  torch.utils.data import Dataset , DataLoader​​​​  

"""
数据加载
"""


类 数据生成器：
    def  __init__ ( self , data_path , config ):
        self.config =配置​​  
        self.path =数据路径 ​ 
        self.vocab = load_vocab  ( config [ " vocab_path" ] ) 
        self.config [ " vocab_size " ] = len ( self.vocab )​​ 
        self.sentences = [  ]​
        self.schema = self.load_schema ( config [ " schema_path " ]  )​ 
        self.load ( )​

    def  load ( self ):
        self.data = [  ]​
        with  open ( self.path , encoding = " utf8 " ) as f : 
            segments  =  f.read ( ). split ( " \ n \n " )
            对于 segments中的每个 segment ： 
                句子 = []
                标签 = []
                for  line  in  segment.split ( " \ n " ) :
                    如果 line.strip ( ) == " "： 
                        继续
                    char , label  =  line.split ( )​
                    句子.追加(字符)
                    labels.append ( self.schema [ label ] )​​​
                self.sentences.append ( " " . join ( sentenece ) )​​
                input_ids  =  self.encode_sentence ( sentenece )​​
                labels  =  self.padding ( labels , -1 )​​​
                self.data.append ( [ torch.LongTensor ( input_ids ) , torch.LongTensor ( labels ) ] )​​​​
        返回

    def  encode_sentence ( self , text , padding = True ):
        input_id  = []
        如果 self.config [ " vocab_path" ] == " words.txt"： 
            for  word  in  jieba.cut ( text ) :​
                input_id.append ( self.vocab.get ( word , self.vocab [ " [ UNK ] " ] ) )​​
        别的：
            对于 文本中的每个字符 ： 
                input_id.append ( self.vocab.get ( char , self.vocab [ " [ UNK ] " ] ) )​​
        如果 填充：
            input_id  =  self.padding ( input_id )​​
        返回 input_id

    #补齐或中断输入的序列，可以在一个批处理内损坏
    def  padding ( self , input_id , pad_token = 0 ):
        input_id  =  input_id [: self.config [ " max_length" ] ]
        input_id  += [ pad_token ] * ( self . config [ "max_length" ] -  len ( input_id ))
        返回 input_id

    def  __len__ ( self ):
        返回 len ( self.data )​​

    def  __getitem__ (自身,索引):
        返回 self.data [ index ]​​

    def  load_schema ( self , path ):
        with  open ( path , encoding = "utf8" ) as  f :
            返回 json.load ( f )​​

#加载字表或词表
def  load_vocab ( vocab_path ):
    token_dict  = {}
    with  open ( vocab_path , encoding = "utf8" ) as  f :
        对于 索引，行在 enumerate  ( f ) :
            token  =  line.strip ( )​
            token_dict[token] = index + 1 #0 移动padding位置，所以从1开始
    返回 token_dict

#用torch自带的DataLoader类封装数据
def  load_data ( data_path , config , shuffle = True ):
    dg = DataGenerator（数据路径，配置）
    dl  =  DataLoader ( dg , batch_size = config [ "batch_size" ], shuffle = shuffle )
    返回 dl



如果 __name__ == "__main__":
    from  config  import  Config
    dg = DataGenerator("../ner_data/train.txt", 配置)
