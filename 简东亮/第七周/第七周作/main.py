# -*- coding: utf-8 -*-

"""
主文件主要任务：参数拼接、模型训练、结果对比和输出模型
"""

导入 torch
导入 时间
导入 numpy 库 并将其命名为 np
导入 操作系统
from  config  import  Config
from  logHandler  import  logger
from  loader  import  load_train_data , load_vocab
从 模型 导入 TorchModel
from  evaluateor  import  Evaluator

logger  =  logger ()
gpu_usable  =  torch.cuda.is_available ( )​​​


def  main ( config ):
    # 加载数据、模型和优化器
    batch_data  =  load_train_data ( config )
    model  =  TorchModel ( config )
    optim  =  torch.optim.Adam ( model.parameters ( ) , lr = config [ " learning_rate " ] )​
    如果 gpu_usable：
        记录器。信息( f"====GPU 可用====" )
        model  =  model.cuda ( )​
    # 开始训练
    记录器。info ( f"====开始训练{ config [ "model_name" ] }模型====" )
    print ( f"====开始训练{ config [ "model_name" ] }模型====" )
    开始时间 = 时间.时间()
    模型.训练()
    for  i  in  range ( config [ "epoch" ]):
        watch_loss  = []
        对于 索引，数据 在 enumerate ( batch_data ):
            如果 gpu_usable：
                data  = [ d . cuda () for  d  in  data ]
            batch_x，batch_y  = 数据
            loss  =  model ( batch_x , batch_y )
            损失.向后()
            优化.步骤()
            优化.零梯度()
            watch_loss.append ( loss.item ( ) )​​
        #记录损失信息
        记录器。info ( f"第{ i + 1 }轮训练结束，该轮平均损失值为：{ np .mean ( watch_loss ) } " )
        print ( f "第{ i + 1 }轮训练结束，该轮平均损失值为：{ np .mean ( watch_loss ) } " )
    # 记录训练运行情况
    执行 时间=  time.time ( ) -  start
    配置[ "执行时间" ] = 执行时间
    # 使用测试集预测，记录预测速率、准确率
    评估器 = 评估器（配置，模型）
    评估器.预测()
    # 保存模型
    如果 配置[ "save_model" ]：
        model_name  =  config [ "model_name" ] +  ".pth"
        model_path  =  os.path.join ( config [ " model_path " ] , model_name )​
        torch.save ( model.state_dict ( ) , model_path )​​
    # 记录模型参数
    记录器。info ( f"模型参数：{ config } " )
    # print(f"模型参数：{config}")


如果 __name__  ==  "__main__" :
    vocab_size  =  load_vocab ( Config [ "vocab_path" ])
    配置[ "vocab_size" ] =  vocab_size
    # 使用不同的模型和超参数来训练，对比结果
    model_type_list  = [ "bert" ]
    # model_type_list = ["LSTM", "RNN", "CNN", "bert"]
    # model_type_list = ["fastText", "RNN", "CNN", "TextCNN", "RCNN", "LSTM", "bert", "bertRNN"]
    num_layers_list  = [ 1 , 3 ]
    双向列表 = [ True , False ]
    lr_list  = [ 1e-3 , 1e-4 ]
    batch_size_list  = [ 20 , 40 ]
    hidden_​​size_list  = [ 256 , 512 ]
    out_channels_list  = [ 64 , 128 ]
    pooling_type_list  = [ "max" , "avg" ]
    对于 model_type_list中的每个 model ： 
        计数 =  0
        配置[ "model_type" ] = 模型
        # 如果是普通的fastText模型，则只是比较不同学习率和batch_size下的模型
        如果 模型 ==  "fastText"：
            对于 lr_list中的每个 lr ： 
                配置[ "学习率" ] =  lr
                for  batch_size  in  batch_size_list :
                    计数 +=  1
                    配置[ "model_name" ] =  model  +  "_"  +  str ( count )
                    配置[ "batch_size" ] =  batch_size
                    main（配置）
        # CNN，只比较feature_dim、hidden_​​size、out_channels
        elif  model  ==  "CNN" :
            for  hidden_​​size  in  hidden_​​size_list :
                配置["hidden_​​size"] = hidden_​​size
                对于 out_channels_list 中的每个 out_channels：
                    配置["out_channels"] = out_channels
                    对于 pooling_type_list 中的每个 pooling_type：
                        计数 += 1
                        配置[ "model_name" ] =  model  +  "_"  +  str ( count )
                        配置["pooling_type"] = pooling_type
                        main（配置）
        # TextCNN，只比较有层数和out_channels
        elif  model  ==  "TextCNN" :
            对于 num_layers_list 中的每个 num_layers：
                配置["num_layers"] = num_layers
                对于 out_channels_list 中的每个 out_channels：
                    计数 += 1
                    配置[ "model_name" ] =  model  +  "_"  +  str ( count )
                    配置["out_channels"] = out_channels
                    main（配置）
        # bert，只比较num_layers
        elif  model  ==  "bert" :
            对于 num_layers_list 中的每个 num_layers：
                计数 += 1
                配置[ "model_name" ] =  model  +  "_"  +  str ( count )
                配置["num_layers"] = num_layers
                main（配置）
        # 比较不同feature_dim、hidden_​​size和双向下的模型
        elif  model  in [ "RNN" , "RCNN" , "LSTM" , "bertRNN" ]:
            for hidden_​​size in hidden_​​size_list:
                配置["hidden_​​size"] = hidden_​​size
                对于双向列表中的每个双向：
                    计数 += 1
                    配置[ "model_name" ] =  model  +  "_"  +  str ( count )
                    配置["双向"] = 双向
                    main（配置）
