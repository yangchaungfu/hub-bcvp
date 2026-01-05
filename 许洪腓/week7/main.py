import torch 
import os 
import random 
import numpy as np 
import logging 
import time 
import pandas as pd
from config import Config 
from model import TorchModel, choose_optimizer
from evaluate import Evaluator 
from loader import load_data 
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 让程序随机行为可复现
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config, is_train=True)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            optimizer.zero_grad()
            input_ids, labels = batch_data 
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)   

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    columns = ["model_type","learning_rate","hidden_size","batch_size","pooling_style","accuracy","time_consuming"]
    results = []
    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn", 'stack_gated_cnn', 'lstm' , 'fast_text']:
    # for model in ["cnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        start_time = time.time()
                        acc = main(Config)
                        time_consuming = time.time() - start_time
                        result = {
                            "model_type": model,
                            "learning_rate": lr,
                            "hidden_size": hidden_size,
                            "batch_size": batch_size,
                            "pooling_style": pooling_style,
                            "accuracy": f"{acc:.2%}",
                            "time_consuming": f"{round(time_consuming,2)}s"
                        }
                        results.append(result)

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(Config["result_path"],index=False)
    print(f"结果已保存到：{Config["result_path"]}")



