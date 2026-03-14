# -*- coding: utf-8 -*-
"""
仅使用 Python 标准库实现一个简化版的 BPE（Byte Pair Encoding）训练器。
 
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional


def clean_corpus_text(text: str) -> str:
    """
    数据清洗
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        raw = line.strip()
        if not raw:
            cleaned_lines.append("")
            continue

        low = raw.lower()
        if "downbank" in low or "下载银行" in raw or "www." in low:
            continue
        if re.fullmatch(r"[-=—_]{3,}", raw):
            continue

        raw = re.sub(r"\s+", " ", raw)
        cleaned_lines.append(raw)

    out_lines: List[str] = []
    blank = False
    for line in cleaned_lines:
        if line == "":
            if not blank:
                out_lines.append("")
            blank = True
        else:
            out_lines.append(line)
            blank = False

    return "\n".join(out_lines).strip() + "\n"


def iter_training_units(text: str, max_units: Optional[int] = None) -> Iterable[str]:
    """
    产生“训练单位”（类似 word 的概念）。
    - 按行产生 unit（中文无空格时更稳）
    """
    count = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        yield line
        count += 1
        if max_units is not None and count >= max_units:
            break


def _get_pair_counts(token_seqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for tokens, freq in token_seqs.items():
        if len(tokens) < 2:
            continue
        prev = tokens[0]
        for cur in tokens[1:]:
            pair_counts[(prev, cur)] += freq
            prev = cur
    return pair_counts


def _merge_pair_in_tokens(tokens: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    a, b = pair
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return tuple(out)


@dataclass
class BPETrainConfig:
    vocab_size: int = 50
    add_special_tokens: bool = True
    special_tokens: Tuple[str, ...] = ("<pad>", "<unk>")
    max_training_units: Optional[int] = 20000


@dataclass
class BPEModel:
    vocab: Dict[str, int]
    id_to_token: List[str]
    merges: List[Tuple[str, str]]


class BPETrainer:
    END_SYMBOL = "</w>"

    def __init__(self, config: BPETrainConfig):
        self.config = config

    def train_from_text(self, text: str) -> BPEModel:
        cleaned = clean_corpus_text(text)

        # 1) 统计字符频次，用于构建“受限初始字母表”
        char_counter: Counter[str] = Counter()
        units: List[str] = []
        for unit in iter_training_units(cleaned, max_units=self.config.max_training_units):
            units.append(unit)
            char_counter.update(list(unit))

        # 2) 计算初始字母表大小：大致取“目标词表的一半”留给 merges
        specials_n = len(self.config.special_tokens) if self.config.add_special_tokens else 0
        target_non_special = max(1, self.config.vocab_size - specials_n)

        base_symbols = max(10, target_non_special // 2)  # 含 </w>
        base_char_slots = max(1, base_symbols - 1)       # 除去 </w>

        most_common_chars = [c for c, _ in char_counter.most_common(base_char_slots)]
        alphabet = set(most_common_chars)
        alphabet.add(self.END_SYMBOL)

        # 3) 构建 token 序列频次：不在 alphabet 的字符统一映射成 <unk>
        unk = "<unk>"
        token_seqs: Dict[Tuple[str, ...], int] = Counter()
        for unit in units:
            tokens: List[str] = []
            for ch in unit:
                tokens.append(ch if ch in alphabet else unk)
            tokens.append(self.END_SYMBOL)
            token_seqs[tuple(tokens)] += 1

        # 4) 初始化 vocab：special tokens + alphabet
        vocab: Dict[str, int] = {}
        id_to_token: List[str] = []

        if self.config.add_special_tokens:
            for t in self.config.special_tokens:
                if t not in vocab:
                    vocab[t] = len(id_to_token)
                    id_to_token.append(t)

        for t in sorted(alphabet):
            if t not in vocab:
                vocab[t] = len(id_to_token)
                id_to_token.append(t)

        merges: List[Tuple[str, str]] = []

        # 5) BPE 迭代合并，直到达到指定 vocab_size
        while len(vocab) < self.config.vocab_size:
            pair_counts = _get_pair_counts(token_seqs)
            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda x: x[1])
            if best_count < 2:
                break

            merges.append(best_pair)

            new_token_seqs: Dict[Tuple[str, ...], int] = Counter()
            for tokens, freq in token_seqs.items():
                merged = _merge_pair_in_tokens(tokens, best_pair)
                new_token_seqs[merged] += freq
            token_seqs = new_token_seqs

            new_token = best_pair[0] + best_pair[1]
            if new_token not in vocab:
                vocab[new_token] = len(id_to_token)
                id_to_token.append(new_token)

            if len(vocab) >= self.config.vocab_size:
                break

        # 6) 如果因为语料太短/可合并对太少导致未达标，则用占位 token 补齐到严格 vocab_size
        extra_i = 0
        while len(vocab) < self.config.vocab_size:
            extra = f"<extra_{extra_i}>"
            if extra not in vocab:
                vocab[extra] = len(id_to_token)
                id_to_token.append(extra)
            extra_i += 1

        return BPEModel(vocab=vocab, id_to_token=id_to_token, merges=merges)

    def save_model(self, model: BPEModel, path: str) -> None:
        obj = {
            "vocab_size": len(model.vocab),
            "vocab": model.vocab,
            "merges": [list(p) for p in model.merges],
            "end_symbol": self.END_SYMBOL,
            "special_tokens": list(self.config.special_tokens) if self.config.add_special_tokens else [],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
