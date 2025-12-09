# -*- coding: utf-8 -*-

import logging
import sys
import os
from config import Config


def logger():
    # 获取引用者的文件名
    caller_file  = sys._getframe(1).f_code.co_filename
    file_name = os.path.splitext(os.path.basename(caller_file))[0]
    # 创建Logger
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)
    # 创建一个文件处理器，并明确指定utf-8编码
    file_handler = logging.FileHandler(Config["log_path"] + "/" + file_name + ".log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    # 设置格式并应用
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    logger = logger()
    logger.info("logHandler初始化完成！！！")