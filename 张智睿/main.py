# main.py
import os
import torch
import logging

from config import Config
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"], config, shuffle=True)

    model = TorchModel(config)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    best_macro = -1.0
    for epoch in range(config["epoch"]):
        model.train()
        logger.info("epoch %d begin" % epoch)

        for input_ids, attention_mask, token_type_ids, labels, word_ids, token_len in train_data:
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()

        # 每轮评估
        evaluator.eval(epoch)

        # 保存：每轮都存一份（也可以只存 best）
        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)

    logger.info("train done.")


if __name__ == "__main__":
    main(Config)