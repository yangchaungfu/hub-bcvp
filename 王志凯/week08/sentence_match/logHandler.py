# -*- coding:utf-8 -*-

import logging
import os
from config import Config


def logger(file_name=None):
    if file_name is None:
        file_name = os.path.basename(__file__)
    log_name = os.path.splitext(file_name)[0] + '.log'
    # 创建Logger
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)
    # 创建一个文件处理器，指定utf8编码
    file_handler = logging.FileHandler(Config["log_base_path"] + "/" + log_name, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    # 设置格式并应用
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    logger = logger()
    logger.info("logHandler初始化完成！！！")