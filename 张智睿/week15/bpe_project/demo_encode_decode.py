# -*- coding: utf-8 -*-
"""
演示 encode / decode 两个功能：
- 输入一句中文，输出 token 序列（id 列表）
- 输入 token 序列，输出对应中文

"""

from __future__ import annotations

import os

from tokenizer import BPETokenizer


def main():
    model_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    model_path = os.path.abspath(model_path)

    tokenizer = BPETokenizer.load(model_path)

    # 可以把这里改成任意一句中文
    text = ("长安到底长什么样")
    ids = tokenizer.encode(text)

    print("原句：", text)
    print("编码得到 token id 序列：")
    print(ids)

    recovered = tokenizer.decode(ids)
    print("\n解码还原：", recovered)


if __name__ == "__main__":
    main()
