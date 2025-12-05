# -*- coding: utf-8 -*-

"""
loader主要任务：加载数据、处理数据、数据封装
"""

导入 torch
导入 pandas  as  pd
from  torch.utils.data import Dataset , DataLoader​​​​  
from  transformers  import  BertTokenizer
from  config  import  Config
from  logHandler  import  logger

logger  =  logger ()
tokenizer  =  BertTokenizer.from_pretrained ( Config [ " pretrain_model_path" ] )


类 数据生成器（数据集）：
    def  __init__ ( self , config , data_type = "train" ):
        super ().__ init__ ()
        self.config =配置​​  
        self.data_path = config [ "  train_data_path " ] 
        self.max_length = config [ " max_length " ]  
        self.total_rows = 0​​  
        self.data = [  ]​
        self.load ( data_type )​​

    def  load ( self , data_type ):
        记录器。info ( "开始加载数据" )
        # 获取总行数
        self.total_rows = pd.read_csv ( self.data_path , usecols = [ 0 ] ) . shape [ 0 ] ​​​​ 
        如果 data_type  ==  "train" :
            # 训练数据只读取80%
            df  =  pd.read_csv ( self.data_path , nrows = int ( self.total_rows * 0.8 ) )​​​​​  
        别的：
            # 测试读取数据剩余20%
            start_index  =  self.total_rows - int ( self.total_rows * 0.2 )​​ ​​   
            # 保留第一行（标题行）
            skip_index_list  =  list ( range ( 1 , start_index  +  1 ))
            df  =  pd.read_csv ( self.data_path , skiprows = skip_index_list )​​​​
        for  row  in  df.itertuples ( ) :
            # 将文本转化为序列
            sequence  =  padding ( self.config [ " model_type " ] , self.max_length , row.review )​​​
            label  =  torch.LongTensor ( [ row.label ] )​​
            self.data.append ( [ sequence , label ] )​​
        记录器。info ( f"数据加载完成，总行数：{ self .total_rows } " )

    def  __len__ ( self ):
        返回 len ( self.data )​​

    def  __getitem__ (自身,索引):
        返回 self.data [ index ]​​


# 将文本转化为序列
def  padding ( model_type , max_len , data ):
    data  =  data [: max_len ]
    如果 model_type  ==  "bert" :
        # bert 模型需要主要加cls和sep
        序列 = 分词器（数据，
                             padding = 'max_length' ,   # 补齐到最大长度
                             truncation = True ,        # 最大最大长度就进行截断
                             max_length = max_len ,     # 最大长度数值
                             return_tensors = "pt" )    #以张量形式输出，后续就需要额外转类型
    别的：
        # tokenizer()返回：input_ids, attention_mask, token_type_ids；
        # 而tokenizer.encode()返回单一的嵌入序列
        sequence  =  tokenizer.encode ( data ,​​
                                    padding = 'max_length' ,
                                    截断= True，
                                    max_length = max_len，
                                    return_tensors = "pt" ,
                                    add_special_tokens = False )   # 是否使用cls sep特殊token(只有bert需要)
    返回 序列


# 加载训练数据
def  load_train_data ( config , shuffle = True ):
    dg  =  DataGenerator ( config )
    train_data  =  DataLoader ( dg , batch_size = config [ "batch_size" ], shuffle = shuffle )
    返回 训练数据


# 加载测试数据
def  load_valid_data ( config ):
    dg  =  DataGenerator（配置，data_type = “有效”）
    valid_data  =  DataLoader ( dg , batch_size = 100 , shuffle = False )
    返回 有效数据


# 加载词表大小
def  load_vocab ( vocab_path ):
    with  open ( vocab_path , "r" , encoding = "utf-8" ) as  f :
        # 这里是为了给原始配置参数，所以只能用原始配置
        返回 sum ( 1  for  _  in  f )


如果 __name__  ==  "__main__" :
    # 初始化数据加载器
    加载器 = 数据生成器（配置）
    记录器。info ( f"DataLoader初始化完成！！！{ loader } " )
