# -*- coding: utf-8 -*-

导入 torch
导入操作系统
导入随机数
导入 numpy 库并将其命名为 np
导入日志
from  config  import  Config
from  model  import  TorchModel , choose_optimizer
从 evaluate 导入 Evaluator
from  loader  import  load_data

logging.basicConfig ( level = logging.INFO , format = ' %(asctime)s - %( name )s - %(levelname)s - %(message) s ' )    
logger  =  logging.getLogger ( __ name__ )​

"""
模型训练主程序
"""

def  main ( config ):
    #创建保存模型的目录
    如果不是 os.path.isdir(config["model_path"]):
        os.mkdir ( config [ "model_path " ] )
    #加载训练数据
    train_data  =  load_data ( config [ "train_data_path" ], config )
    #加载模型
    model  =  TorchModel ( config )
    # 标识是否使用gpu
    cuda_flag  =  torch.cuda.is_available ( )​​​
    如果 cuda_flag：
        记录器。info ( "gpu可以使用，迁移模型至gpu" )
        model  =  model.cuda ( )​
    #加载优化器
    优化器 = choose_optimizer(配置, 模型)
    #加载效果测试类
    评估器 = 评估器(配置、模型、日志记录器)
    # 训练
    for  epoch  in  range ( config [ "epoch" ]):
        epoch  +=  1
        模型.训练()
        logger.info ( " epoch %d begin " % epoch )  
        train_loss  = []
        for  index , batch_data  in  enumerate ( train_data ):
            优化器.zero_grad()
            如果 cuda_flag：
                batch_data  = [ d . cuda () for  d  in  batch_data ]
            input_id, labels = batch_data #输入变化时这里需要修改，比如多输入，多输出的情况
            损失 = 模型(输入id, 标签)
            损失.向后()
            优化器.step()
            train_loss.append ( loss.item ( ) )​​
            如果 index % int(len(train_data) / 2) == 0:
                logger.info("批次损失 %f" % 损失)
        logger.info("epoch 平均损失：%f" % np.mean(train_loss))
        评估器.eval(epoch)
    model_path  =  os.path.join ( config [ "model_path" ] , " epoch_% d.pth " % epoch )  
    # torch.save(model.state_dict(), model_path)
    返回模型和训练数据

如果 __name__ == "__main__":
    模型，训练数据 = main(配置)
