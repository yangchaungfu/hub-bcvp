# -*- coding: utf-8 -*-
"""
BPE 分词器：支持 encode/decode，以及从 tokenizer.json 加载。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class BPETokenizer:
    vocab: Dict[str, int]
    id_to_token: List[str]
    merges: List[Tuple[str, str]]
    end_symbol: str = "</w>"
    unk_token: str = "<unk>"

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        vocab = obj["vocab"]
        id_to_token = [None] * len(vocab)
        for tok, idx in vocab.items():
            id_to_token[idx] = tok

        merges = [tuple(x) for x in obj.get("merges", [])]
        end_symbol = obj.get("end_symbol", "</w>")

        # unk_token：默认 <unk>，若 vocab 里没有则退化为任意一个 token
        unk_token = "<unk>" if "<unk>" in vocab else id_to_token[0]

        return cls(
            vocab=vocab,
            id_to_token=id_to_token,
            merges=merges,
            end_symbol=end_symbol,
            unk_token=unk_token
        )

    def save(self, path: str) -> None:
        obj = {
            "vocab_size": len(self.vocab),
            "vocab": self.vocab,
            "merges": [list(p) for p in self.merges],
            "end_symbol": self.end_symbol,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _apply_bpe(self, symbols: List[str]) -> List[str]:
        """
        对单个 symbol 序列应用 merges。
        因为本项目词表很小（50），merges 数量也很少，直接按顺序做替换即可。
        """
        for a, b in self.merges:
            merged = a + b
            out: List[str] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def encode(self, text: str, add_end_symbol: bool = True) -> List[int]:
        """
        输入一句中文，输出 token id 序列。
        - 默认会在末尾加入 </w>（与训练一致），让 decode 更稳定
        """
        # 初始符号：逐字符
        symbols = list(text)
        if add_end_symbol:
            symbols.append(self.end_symbol)

        bpe_tokens = self._apply_bpe(symbols)

        ids: List[int] = []
        for t in bpe_tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            else:
                ids.append(self.vocab.get(self.unk_token, 0))
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        输入 token id 序列，输出对应中文。
        - 把 token 拼接后，移除 </w> 边界符
        """
        tokens: List[str] = []
        for i in ids:
            if 0 <= i < len(self.id_to_token):
                tokens.append(self.id_to_token[i])
            else:
                tokens.append(self.unk_token)

        text = "".join(tokens)
        # 去掉边界符（如果出现多次也一并去掉）
        text = text.replace(self.end_symbol, "")
        return text
