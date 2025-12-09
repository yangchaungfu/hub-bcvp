import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data, split_csv_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 存储所有实验结果
results = []

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config, verbose=True):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = TorchModel(config)

    # 检测设备：优先CUDA，其次MPS，最后CPU
    if torch.cuda.is_available():
        device = "cuda"
        if verbose:
            logger.info("CUDA可以使用，迁移模型至GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        if verbose:
            logger.info("MPS可以使用，迁移模型至MPS")
    else:
        device = "cpu"
        if verbose:
            logger.info("使用CPU训练")
    
    # 迁移模型到对应设备
    model = model.to(device)
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类（验证集）
    evaluator = Evaluator(config, model, logger, data_type="valid", device=device)

    # 训练
    best_valid_acc = 0.0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        if verbose:
            logger.info("开始第 %d 轮训练" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            # 将数据迁移到对应设备
            batch_data = [d.to(device) for d in batch_data]

            optimizer.zero_grad()
            # 处理BERT和非BERT模型的不同输入格式
            if config["model_type"] in ["bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]:
                input_ids, attention_mask, labels = batch_data
                loss = model(input_ids, labels, attention_mask=attention_mask)
            else:
                input_ids, labels = batch_data
                loss = model(input_ids, labels)
            
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().item())
            if verbose and index % max(1, int(len(train_data) / 2)) == 0:
                logger.info("batch loss %f" % loss.detach().item())

        if verbose:
            logger.info("epoch average loss: %f" % np.mean(train_loss))

        # 在验证集上评估
        valid_acc = evaluator.eval(epoch)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

    # 在测试集上评估最终效果
    test_evaluator = Evaluator(config, model, logger, data_type="test", device=device)
    test_acc = test_evaluator.eval(epoch)

    return test_acc


def format_config_for_table(config):
    """
    格式化配置信息用于表格显示
    
    Args:
        config: 配置字典
    
    Returns:
        格式化的参数字符串
    """
    model_type = config["model_type"]
    params = []

    if model_type == "lstm" or model_type == "gru" or model_type == "rnn":
        params.append(f"{config['hidden_size']}神经元")
        params.append(f"{config['num_layers']}层")
    elif model_type == "bert":
        params.append(f"{config['num_layers']}层")
    elif model_type in ["cnn", "gated_cnn", "stack_gated_cnn", "rcnn"]:
        params.append(f"卷积核{config['kernel_size']}")
        params.append(f"{config['hidden_size']}隐藏层")

    params.append(f"学习率{config['learning_rate']}")
    params.append(f"批大小{config['batch_size']}")
    params.append(f"{config['pooling_style']}池化")

    return " | ".join(params)


if __name__ == "__main__":
    """
    训练出一个，输入一句话，能预测出结果的模型
    1. 数据分为训练集、验证集和测试集（7:2:1）
    2. 数据分析（文本长度，指定一个最大长度） 

    输出一个表格：
    模型，参数，效果是多少 
    lstm | 256神经元 | 2层 | 学习率0.001 | 准确率 60%
    bert | 3层 | 学习率0.001 | 准确率 80%
    """

    # 首先检查并分割CSV数据
    csv_path = "文本分类练习.csv"
    data_dir = "./data"

    if not os.path.exists(os.path.join(data_dir, "train_tag_news.json")):
        logger.info("=" * 60)
        logger.info("开始分割CSV数据...")
        logger.info("=" * 60)
        try:
            split_csv_data(csv_path, output_dir=data_dir, random_seed=Config["seed"])
            # 更新配置路径
            Config["train_data_path"] = os.path.join(data_dir, "train_tag_news.json")
            Config["valid_data_path"] = os.path.join(data_dir, "valid_tag_news.json")
            Config["test_data_path"] = os.path.join(data_dir, "test_tag_news.json")
        except Exception as e:
            logger.error(f"数据分割失败: {str(e)}")
            raise
    else:
        logger.info("数据文件已存在，跳过分割步骤")
        Config["train_data_path"] = os.path.join(data_dir, "train_tag_news.json")
        Config["valid_data_path"] = os.path.join(data_dir, "valid_tag_news.json")
        Config["test_data_path"] = os.path.join(data_dir, "test_tag_news.json")

    # 对比所有模型
    # models_to_test = ["bert", "lstm", "cnn"]
    models_to_test = ["bert"]

    logger.info("=" * 80)
    logger.info("开始模型训练和评估")
    logger.info("=" * 80 + "\n")

    total_experiments = 0
    for model in models_to_test:
        Config["model_type"] = model

        # 根据模型类型设置不同的超参数搜索空间
        if model == "bert":
            lr_list = [1e-3, 1e-4]
            hidden_size_list = [256]  # BERT使用预训练模型的hidden_size
            batch_size_list = [32, 64]
            pooling_list = ["avg", "max"]
        elif model == "lstm":
            lr_list = [1e-3, 1e-4]
            hidden_size_list = [128, 256]
            batch_size_list = [64, 128]
            pooling_list = ["avg", "max"]
        elif model == "cnn":
            lr_list = [1e-3, 1e-4]
            hidden_size_list = [128, 256]
            batch_size_list = [64, 128]
            pooling_list = ["avg", "max"]

        for lr in lr_list:
            Config["learning_rate"] = lr
            for hidden_size in hidden_size_list:
                Config["hidden_size"] = hidden_size
                for batch_size in batch_size_list:
                    Config["batch_size"] = batch_size
                    for pooling_style in pooling_list:
                        Config["pooling_style"] = pooling_style
                        total_experiments += 1

    logger.info(f"总共需要训练 {total_experiments} 个模型配置\n")

    experiment_count = 0
    for model in models_to_test:
        Config["model_type"] = model

        # 根据模型类型设置不同的超参数搜索空间
        if model == "bert":
            lr_list = [1e-3, 1e-4]
            hidden_size_list = [256]  # BERT使用预训练模型的hidden_size
            batch_size_list = [32, 64]
            pooling_list = ["avg", "max"]
        elif model == "lstm":
            lr_list = [1e-3, 1e-4]
            hidden_size_list = [128, 256]
            batch_size_list = [64, 128]
            pooling_list = ["avg", "max"]
        elif model == "cnn":
            lr_list = [1e-3, 1e-4]
            hidden_size_list = [128, 256]
            batch_size_list = [64, 128]
            pooling_list = ["avg", "max"]

        for lr in lr_list:
            Config["learning_rate"] = lr
            for hidden_size in hidden_size_list:
                Config["hidden_size"] = hidden_size
                for batch_size in batch_size_list:
                    Config["batch_size"] = batch_size
                    for pooling_style in pooling_list:
                        Config["pooling_style"] = pooling_style
                        experiment_count += 1

                        logger.info(f"{'=' * 80}")
                        logger.info(f"实验 {experiment_count}/{total_experiments}")
                        logger.info(f"模型: {model}")
                        logger.info(f"配置: {Config}")
                        logger.info(f"{'=' * 80}\n")

                        try:
                            # 训练时只显示关键信息，减少日志输出
                            accuracy = main(Config, verbose=(experiment_count <= 3))
                            params_str = format_config_for_table(Config)
                            results.append({
                                "模型": model.upper(),
                                "参数": params_str,
                                "准确率": f"{accuracy * 100:.2f}%",
                                "准确率数值": accuracy  # 用于排序
                            })
                            logger.info(f"✓ 模型 {model} 训练完成，测试集准确率: {accuracy * 100:.2f}%")
                        except Exception as e:
                            logger.error(f"✗ 模型 {model} 训练出错: {str(e)}")
                            import traceback

                            traceback.print_exc()
                            continue

    # 输出结果表格
    if results:
        # 按准确率排序
        sorted_results = sorted(results, key=lambda x: x["准确率数值"], reverse=True)

        print("\n" + "=" * 100)
        print("实验结果汇总表")
        print("=" * 100)
        print(f"{'模型':<10} {'参数':<60} {'准确率':<10}")
        print("-" * 100)
        for r in sorted_results:
            print(f"{r['模型']:<10} {r['参数']:<60} {r['准确率']:<10}")
        print("=" * 100)

        # 保存到文件
        output_file = "实验结果.txt"
        with open(output_file, "w", encoding="utf8") as f:
            f.write("实验结果汇总表\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'模型':<10} {'参数':<60} {'准确率':<10}\n")
            f.write("-" * 100 + "\n")
            for r in sorted_results:
                f.write(f"{r['模型']:<10} {r['参数']:<60} {r['准确率']:<10}\n")
        print(f"\n✓ 结果已保存到: {output_file}")

        # 显示最佳结果
        if len(sorted_results) > 0:
            best = sorted_results[0]
            print(f"\n最佳模型:")
            print(f"  模型: {best['模型']}")
            print(f"  参数: {best['参数']}")
            print(f"  准确率: {best['准确率']}")
    else:
        logger.warning("没有成功完成任何实验！")
