# -*- coding: utf-8 -*-
"""
build_tokenizer.py
- 读取 data/语料
- 训练 BPE 词表到 50 个 token
- 输出 tokenizer.json（包含词表与 token-id 对应关系、merges）
"""

from __future__ import annotations

import os

from bpe_trainer import BPETrainer, BPETrainConfig


def main():
  
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "长安乱.txt")
    corpus_path = os.path.abspath(corpus_path)

    text = None
    for enc in ("utf-8", "gb18030"):
        try:
            with open(corpus_path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        # 最后兜底：忽略错误读取
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    config = BPETrainConfig(
        vocab_size=50,                 # 按你的要求：50 词规模词表
        add_special_tokens=True,       # 词表里包含 <pad> <unk>
        max_training_units=20000       # 小说很长，这个值够学到高频模式
    )
    trainer = BPETrainer(config)
    model = trainer.train_from_text(text)

    out_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    out_path = os.path.abspath(out_path)
    trainer.save_model(model, out_path)

    print("训练完成！")
    print("词表大小 vocab_size =", len(model.vocab))
    print("配置已保存到：", out_path)

    vocab_items = sorted(model.vocab.items(), key=lambda x: x[1])
    print("\n前 20 个 token：")
    for tok, idx in vocab_items[:20]:
        print(f"{idx:>3}  {tok}")


if __name__ == "__main__":
    main()
