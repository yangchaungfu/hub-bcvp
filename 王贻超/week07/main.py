# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序（支持多模型对比）
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    best_acc = 0
    best_pred_time = 0
    
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc, pred_time = evaluator.eval(epoch)
        
        # 记录最佳结果
        if acc > best_acc:
            best_acc = acc
            best_pred_time = pred_time

            # 保存最佳模型
            model_path = os.path.join(config["model_path"], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            
    return best_acc, best_pred_time

if __name__ == "__main__":
    # 先运行数据预处理
    from preprocess_data import analyze_data, convert_csv_to_json
    analyze_data('文本分类练习.csv')
    convert_csv_to_json('文本分类练习.csv')
    
    # 模型对比实验
    results = []
    
    # 获取所有可用的模型类型
    all_models = [
        "fast_text", "cnn", "lstm", "gru", "rnn",
        "gated_cnn", "stack_gated_cnn", "rcnn",
        "bert", "bert_lstm", "bert_cnn", "bert_mid_layer"
    ]
    
    for model_type in all_models:
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_type}")
        print(f"{'='*60}")
        
        # 更新配置
        Config["model_type"] = model_type
        Config["model_path"] = f"review_output_{model_type}"
        
        # 根据模型类型调整超参数
        if model_type == "bert":
            Config["learning_rate"] = 1e-5
            Config["batch_size"] = 32  # BERT内存消耗大，减小batch size
        elif "bert" in model_type:
            Config["learning_rate"] = 1e-4
            Config["batch_size"] = 32
        else:
            Config["learning_rate"] = 1e-3
            Config["batch_size"] = 64
            
        try:
            acc, pred_time = main(Config)
            results.append({
                "model": model_type,
                "accuracy": acc,
                "prediction_time_ms": pred_time
            })
            print(f"模型 {model_type} 训练完成:")
            print(f"  最佳准确率: {acc:.4f}")
            print(f"  平均预测时间: {pred_time:.4f} ms/sample")
        except Exception as e:
            print(f"模型 {model_type} 训练出错: {e}")
            results.append({
                "model": model_type,
                "accuracy": 0,
                "prediction_time_ms": 0
            })
    
    # 输出结果表格
    print("\n" + "="*70)
    print("电商评论情感分类模型对比结果")
    print("="*70)
    print(f"{'模型类型':<20} {'准确率':<15} {'平均预测时间(ms)':<20}")
    print("-"*70)
    for result in results:
        print(f"{result['model']:<20} {result['accuracy']:<15.4f} {result['prediction_time_ms']:<20.4f}")
    
    # 保存结果到文件
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results.to_csv("all_model_comparison_results.csv", index=False, encoding='utf-8-sig')
    print("\n结果已保存到 all_model_comparison_results.csv 文件中")
