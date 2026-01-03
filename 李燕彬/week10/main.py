# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import BertLanguageModel, choose_optimizer
from evaluate import Evaluator
from loader import build_dataset, load_vocab, load_corpus

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
语音预测模型主程序
"""



def main(save_weight=True):

    #加载模型
    model = BertLanguageModel(Config)

    # 标识是否使用cuda
    use_cuda = torch.cuda.is_available()
    print(f"是否使用cuda：{use_cuda}")
    if use_cuda:
        model = model.cuda()

    # 获取配置参数, 初始化训练前的数据
    epoch_num = Config["epoch_num"]  # 训练轮数
    batch_size = Config["batch_size"]  # 每次训练样本个数
    train_sample = Config["train_sample"]  # 每轮训练总共训练的样本总数
    window_size = Config["window_size"] # 样本长度
    text_length = Config["text_length"] # 生成文本长度

    # 加载优化器
    optim = choose_optimizer(Config, model)

    # 加载评估器
    evaluator = Evaluator(Config, model, text_length, logger)


    print("文本词表模型加载完毕，开始训练")
    import sys
    sys.stdout.flush()
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, model.vocab, window_size, model.corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
            if batch_idx % 255 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                sys.stdout.flush()
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        sys.stdout.flush()
        print(evaluator.generate_sentence("让他在半年之前，就不能做出", model, model.vocab, window_size))
        sys.stdout.flush()
        print(evaluator.generate_sentence("李慕站在山路上，深深的呼吸", model, model.vocab, window_size))
        sys.stdout.flush()
    if not save_weight:
        return
    else:
        base_name = os.path.basename(Config["corpus_path"]).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main(False)