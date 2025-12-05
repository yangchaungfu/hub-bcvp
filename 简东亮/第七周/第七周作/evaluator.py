# -*- coding: utf-8 -*-

"""
evaluator任务：对测试集进行测试、比较不同模型不同参数训练结果、将config数据进行输出
"""

导入 torch
导入时间
导入 numpy 库并将其命名为 np
from  logHandler  import  logger
from  loader  import  load_valid_data
logger  =  logger ()

类评估器：
    def  __init__ ( self , config , model ):
        self.config = 配置
        self.model = 模型
        自我效率 = 0
        self.correct_percent = 0​​  

    def  predict ( self ):
        print ( f"===使用模型{ self .config [ "model_name" ] }开始预测=== " )
        batch_data  =  load_valid_data ( self . config )
        # 平均每百条运行量（最后一个批次可能不足100条，要去掉）
        execution_time_avg  = []
        正确率 = 0
        错误 = 0
        self.model.eval ( )​​​
        对于 batch_data 中的每个 batch_x 和 batch_y：
            开始时间 = time.time()
            使用 torch.no_grad()：
                y_pred  =  self.model ( batch_x )​​
                for  y_p , y  in  zip ( y_pred , batch_y ):
                    如果 y_p[int(y[0])] > 0.5：
                        正确 += 1
                    别的：
                        # print(f"预测错误！预测值：{y_p}, 真实值：{y}")
                        错误 += 1
            # 每批运行
            执行时间 = time.time() - 开始时间
            execution_time_avg.append ( execution_time )​​
        self.efficiency = f" { np.mean ( execution_time_avg [ : - 1 ] ) :. 6f } "  
        print(f"每百条预测运行：{self.efficiency}")
        # 计算正确率
        self.correct_percent = f" { correct / ( correct + wrong ):. 4 % } "     
        print ( f" 预测准确率：{ self . Correct_percent } " )
        self.model_config ( )​

    # 整理出模型参数和训练结果
    def  model_config ( self ):
        model_test_info  =  dict ()
        model_test_info [ "model" ] = self.config  [ " model_name " ]
        model_test_info [ "learning_rate" ] = self.config  [ " learning_rate " ]
        model_test_info [ "batch_size" ] = self.config  [ " batch_size " ]
        model_test_info [ "hidden_​​size" ] = self.config  [ " hidden_​​size " ]
        model_test_info [ "out_channels" ] = self.config  [ " out_channels " ]
        model_test_info [ "num_layers" ] = self.config  [ " num_layers " ]
        model_test_info [ "bidirectional" ] = self.config  [ " bidirectional " ]
        model_test_info [ "pooling_type" ] = self.config  [ " pooling_type " ]
        model_test_info [ "效率" ] =  self .效率
        model_test_info [ " correct_percent " ] =  self.correct_percent
        logger.info ( model_test_info )
