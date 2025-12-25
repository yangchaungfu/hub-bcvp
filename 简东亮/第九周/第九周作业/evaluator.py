# -*- coding: utf-8 -*-
导入 torch
导入 re
导入 numpy 库并将其命名为 np
from  collections  import  defaultdict
from  loader  import  load_data

"""
模型效果测试
"""

类评估器：
    def  __init__ ( self , config , model , logger ):
        self.config = 配置
        self.model = 模型
        self.logger = logger​​  
        self.valid_data = load_data  ( config [ "valid_data_path " ] , config , shuffle = False ) 


    def  eval ( self , epoch ):
        自己。记录器。info ( "开始测试第%d轮模型效果："  %  epoch )
        self.stats_dict = { " LOCATION  " : defaultdict ( int ),
                           "TIME" : defaultdict ( int ),
                           “PERSON”：defaultdict(int)，
                           "组织": defaultdict(int)}
        self.model.eval ( )​​​
        for  index , batch_data  in  enumerate ( self.valid_data ) :​
            sentences  =  self.valid_data.dataset.sentences [ index * self.config [ " batch_size " ] :( index + 1 ) * self.config [ " batch_size " ] ]​​​​   
            如果 torch.cuda.is_available():
                batch_data  = [ d . cuda () for  d  in  batch_data ]
            input_id, labels = batch_data #输入变化时这里需要修改，比如多输入，多输出的情况
            使用 torch.no_grad()：
                pred_results = self.model(input_id) #不输入标签，使用模型当前参数进行预测
            self.write_stats ( labels , pred_results , sentences )​​
        self.show_stats ( )​
        返回

    def  write_stats ( self , labels , pred_results , sentences ):
        断言 len(标签) == len(pred_results) == len(句子)
        如果 self.config["use_crf"] 不存在：
            pred_results  =  torch.argmax ( pred_results , dim = -1 )​​​
        对于 zip(labels, pred_results, sentences):
            如果 不是 self.config [ "use_crf " ] ：
                pred_label  =  pred_label.cpu (). detach ( ). tolist ( )
            true_label  =  true_label.cpu (). detach ( ). tolist ( )
            true_entities  =  self.decode ( sentence , true_label )​​
            pred_entities  =  self.decode ( sentence , pred_label )​​
            # print("=+++++++++")
            # print(true_entities)
            # print(pred_entities)
            # print('=+++++++++')
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 识别率 = 识别出的正确实体数 / 样本的实体数
            for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]:
                self.stats_dict [ key ][ "正确识别" ] += len ( [ ent for ent in pred_entities [ key ] if ent in true_entities [ key ] ] )        
                自己。stats_dict [ key ][ "样本实体数" ] +=  len ( true_entities [ key ])
                自己。stats_dict [ key ][ "识别出实体数" ] +=  len ( pred_entities [ key ])
        返回

    def  show_stats ( self ):
        F1_scores  = []
        for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 识别率 = 识别出的正确实体数 / 样本的实体数
            精度 = 自我。stats_dict [ key ][ "正确识别" ] / ( 1e-5  +  self . stats_dict [ key ][ "识别出实体数" ])
            回忆 = 自我。stats_dict [ key ][ "正确识别" ] / ( 1e-5  +  self . stats_dict [ key ][ "样本实体数" ])
            F1  = ( 2  * 精确率 * 召回率) / (精确率 + 召回率 +  1e-5 )
            F1_scores.append ( F1 )​​
            自己。记录器。info ( "%s类实体，准确率：%f,识别率: %f, F1: %f"  % ( key , precision , recall , F1 ))
        self.logger.info ( " Macro -F1: % f " % np.mean ( F1_scores ) )  
        correct_pred  =  sum ([ self.stats_dict [ key ][ "正确识别" ] for key in [ " PERSON " , "LOCATION" , "TIME" , "ORGANIZATION" ]])  
        Total_pred  =  sum ([ self . stats_dict [ key ][ "识别出实体数" ] for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]])
        true_enti  =  sum ([ self . stats_dict [ key ][ "样本实体数" ] for  key  in [ "PERSON" , "LOCATION" , "TIME" , "ORGANIZATION" ]])
        微精度 = 正确预测值 / (总预测值 +  1e-5 )
        微回忆率 = 正确预测值 / (真实值 +  1e-5 )
        micro_f1  = ( 2  *  micro_precision  *  micro_recall ) / ( micro_precision  +  micro_recall  +  1e-5 )
        self.logger.info ( " Micro-F1 % f " % micro_f1 )  
        self.logger.info ( " -------------------- " )​​
        返回

    '''
    {
      “B-位置”：0，
      “B组织”：1，
      “B-PERSON”：2，
      “B-TIME”：3，
      “I-LOCATION”：4，
      “I-组织”：5，
      “I-PERSON”：6，
      “I-TIME”：7，
      “O”：8
    }
    '''
    def  decode ( self , sentence , labels ):
        labels  =  "" .join ( [ str ( x ) for  x  in  labels [: len ( sentence )]])
        results  =  defaultdict ( list )
        for  location  in  re.finditer ( " (04+)" , labels ) :
            s , e  =  location.span ( )​
            results [ "LOCATION" ]. append ( sentence [ s : e ])
        for  location  in  re.finditer ( " (15+)" , labels ) :
            s , e  =  location.span ( )​
            results [ "ORGANIZATION" ]. append ( sentence [ s : e ])
        for  location  in  re.finditer ( " (26+)" , labels ) :
            s , e  =  location.span ( )​
            results [ "PERSON" ]. append ( sentence [ s : e ])
        for  location  in  re.finditer ( " (37+)" , labels ) :
            s , e  =  location.span ( )​
            results [ "TIME" ]. append ( sentence [ s : e ])
        返回 结果
