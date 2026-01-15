# coding:utf8
"""
基于第十周作业nnlm.py 框架，改造成：
1）SFT 形式的 mask（只在“助手回答”部分计算 loss）；
2）seq-to-seq 训练；
3）使用 BERT 作为 encoder + BERT 作为 decoder（Bert2Bert / EncoderDecoderModel）；
4）使用第八周的电商问答预料
"""

import os
import json
import math
import random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn

from transformers import AutoTokenizer, EncoderDecoderModel


# =========================
# 数据读取
# =========================

def _resolve_data_dir(project_root: str) -> str:
    """传项目根目录 or 直接传 data/ 目录"""
    project_root = os.path.abspath(project_root)
    data_dir = os.path.join(project_root, "data")
    return data_dir if os.path.isdir(data_dir) else project_root


def _read_jsonl_or_json(path: str) -> List[Dict]:
    """
    支持：
    - JSONL：每行一个 JSON 对象
    - JSON：list[dict] 或 dict（则当作单条）
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # 粗略判断 JSONL
    if len(lines) >= 2 and all(ln.startswith("{") for ln in lines[: min(5, len(lines))]):
        return [json.loads(ln) for ln in lines]

    obj = json.loads(raw)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _load_schema(data_dir: str) -> Optional[Dict[str, int]]:
    """读取 schema.json: {label: id}"""
    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.isfile(schema_path):
        return None
    with open(schema_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return None
    out = {}
    for k, v in obj.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out if out else None


def load_sft_pairs(project_root: str, max_samples: int = 0) -> List[Tuple[str, str]]:

    data_dir = _resolve_data_dir(project_root)
    schema = _load_schema(data_dir)

    # 优先读取 train.json，其次 data.json（可选）
    files = []
    for name in ["train.json", "train.jsonl", "data.json", "data.jsonl"]:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            files.append(p)

    rows_all: List[Dict] = []
    labels_seen: List[str] = []

    for p in files:
        rows = _read_jsonl_or_json(p)
        rows_all.extend(rows)
        for r in rows:
            tgt = r.get("target")
            if isinstance(tgt, str) and tgt.strip():
                labels_seen.append(tgt.strip())

    if schema:
        # 按 id 排序的标签列表
        label_list = [k for k, _ in sorted(schema.items(), key=lambda kv: kv[1])]
    else:
        label_list = sorted(set(labels_seen))

    def build_instruction(user_text: str) -> str:
        labels = "、".join(label_list)
        return (
            "你是一个意图分类助手。请从下列意图标签中选择最合适的一个，并且只输出标签名，不要输出其它内容。\n"
            f"意图标签：{labels}\n"
            f"用户输入：{user_text}"
        )

    pairs: List[Tuple[str, str]] = []
    for r in rows_all:
        qs = r.get("questions")
        tgt = r.get("target")
        if not (isinstance(qs, list) and isinstance(tgt, str) and tgt.strip()):
            continue
        tgt = tgt.strip()
        for q in qs:
            if not (isinstance(q, str) and q.strip()):
                continue
            pairs.append((build_instruction(q.strip()), tgt))
            if max_samples and len(pairs) >= max_samples:
                return pairs[:max_samples]

    return pairs[:max_samples] if max_samples else pairs


# =========================
# SFT mask + seq2seq batch 构造
# =========================

def _find_subsequence(haystack: List[int], needle: List[int], start_at: int = 0) -> int:
    """返回 needle 在 haystack 中首次出现的位置；找不到返回 -1"""
    if not needle:
        return -1
    for i in range(start_at, max(start_at, len(haystack) - len(needle) + 1)):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def build_sft_decoder_inputs_and_labels(
    tokenizer,
    instruction: str,
    response: str,
    max_target_len: int,
    user_prefix: str = "用户：",
    assistant_prefix: str = "助手：",
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    decoder 序列拼接为：
      [CLS] 用户：{instruction}\n助手：{response} [SEP]

    labels：与 decoder_ids 相同，但 prompt 部分（用户：...助手：）置为 -100，使 loss 只算 response token。
    同时对 padding 也置 -100。

    注意：teacher forcing 要做“右移”：decoder_input_ids[t] 预测 labels[t]（下一 token）
    """
    prompt = f"{user_prefix}{instruction}\n{assistant_prefix}"
    full = f"{prompt}{response}"

    dec = tokenizer(
        full,
        max_length=max_target_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,   # BERT 会加 [CLS] ... [SEP]
    )
    decoder_ids = dec["input_ids"]          # (1, T)
    decoder_attn = dec["attention_mask"]    # (1, T)

    # 找到 prompt 在 full token 序列中的位置（不带 special tokens 的 prompt_ids）
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = decoder_ids[0].tolist()

    start = _find_subsequence(full_ids, prompt_ids, start_at=0)
    if start == -1:
        # 截断等原因可能找不到：保守地 mask 前缀
        start = 1
        prompt_len = min(len(prompt_ids), max(0, max_target_len - 1))
    else:
        prompt_len = len(prompt_ids)

    labels = decoder_ids.clone()
    mask_upto = min(max_target_len, start + prompt_len)
    labels[0, :mask_upto] = -100  # prompt（以及一般包含 [CLS]）不算 loss

    # padding 也不算 loss
    pad_id = tokenizer.pad_token_id
    labels[labels == pad_id] = -100

    # teacher forcing：右移 decoder_input_ids
    decoder_input_ids = decoder_ids.clone()
    decoder_input_ids[:, 1:] = decoder_ids[:, :-1]
    decoder_input_ids[:, 0] = tokenizer.cls_token_id

    return (
        decoder_input_ids.squeeze(0),
        decoder_attn.squeeze(0),
        labels.squeeze(0),
    )


def get_batch(
    pairs: List[Tuple[str, str]],
    tokenizer,
    batch_size: int,
    max_source_len: int,
    max_target_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    - 从 pairs 随机抽 batch_size 条
    - 构造 encoder 输入 + decoder 输入 + labels（含 SFT mask）
    """
    batch = random.sample(pairs, batch_size) if len(pairs) >= batch_size else [random.choice(pairs) for _ in range(batch_size)]

    input_ids, attention_mask = [], []
    decoder_input_ids, decoder_attention_mask, labels = [], [], []

    for instruction, response in batch:
        enc = tokenizer(
            instruction,
            max_length=max_source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )
        dec_in, dec_attn, lab = build_sft_decoder_inputs_and_labels(
            tokenizer, instruction, response, max_target_len
        )

        input_ids.append(enc["input_ids"].squeeze(0))
        attention_mask.append(enc["attention_mask"].squeeze(0))
        decoder_input_ids.append(dec_in)
        decoder_attention_mask.append(dec_attn)
        labels.append(lab)

    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_mask).to(device),
        "decoder_input_ids": torch.stack(decoder_input_ids).to(device),
        "decoder_attention_mask": torch.stack(decoder_attention_mask).to(device),
        "labels": torch.stack(labels).to(device),
    }


# =========================
# 模型构建（用 BERT 训练 seq2seq）
# =========================

def build_model(model_name: str = "bert-base-chinese"):
    """
    使用 EncoderDecoderModel：BERT encoder + BERT decoder（decoder 开 cross-attention）。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

    # 配置关键 token id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 让 decoder 真正成为 decoder
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    return model, tokenizer


@torch.no_grad()
def predict_label(question: str, model, tokenizer, label_list: List[str], device: torch.device) -> str:
    """
    推理：把 question 包装成 instruction，然后 seq2seq generate，取生成结果。
    """
    instruction = (
        "你是一个意图分类助手。请从下列意图标签中选择最合适的一个，并且只输出标签名，不要输出其它内容。\n"
        f"意图标签：{'、'.join(label_list)}\n"
        f"用户输入：{question}"
    )
    enc = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=256).to(device)
    out_ids = model.generate(
        **enc,
        max_length=64,
        num_beams=4,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0].tolist()
    text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    # 如果生成包含“助手：”，只取后半
    m = re.search(r"助手：(.+)$", text, flags=re.S)
    if m:
        text = m.group(1).strip()
    # 尝试匹配到标签（避免模型生成多余文本）
    for lb in label_list:
        if lb == text:
            return lb
    # 兜底：取第一行/第一个词
    return text.splitlines()[0].strip().split()[0]


# =========================
# 训练
# =========================

def train(project_root: str = ".", save_weight: bool = True):
    # 下面这些超参
    epoch_num = 5                 # 训练轮数
    batch_size = 16               # batch size
    train_steps_per_epoch = 2000  # 每个 epoch 的 step
    lr = 5e-5                     # 学习率
    max_source_len = 256          # encoder 最大长度
    max_target_len = 128          # decoder 最大长度
    model_name = "bert-base-chinese"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 读取训练数据
    pairs = load_sft_pairs(project_root)
    if len(pairs) == 0:
        raise RuntimeError("没有在 data/ 目录下找到可用训练样本（train.json/data.json）。请检查数据格式。")

    # label 列表（用于推理展示；训练时已写进 instruction）
    data_dir = _resolve_data_dir(project_root)
    schema = _load_schema(data_dir)
    if schema:
        label_list = [k for k, _ in sorted(schema.items(), key=lambda kv: kv[1])]
    else:
        # 从 pairs 的 response 收集
        label_list = sorted(set([r for _, r in pairs]))

    print(f"加载样本数：{len(pairs)}")
    print(f"标签数：{len(label_list)}")

    # 建模
    model, tokenizer = build_model(model_name)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(epoch_num):
        losses = []
        for step in range(train_steps_per_epoch):
            batch = get_batch(
                pairs, tokenizer,
                batch_size=batch_size,
                max_source_len=max_source_len,
                max_target_len=max_target_len,
                device=device,
            )
            out = model(**batch)
            loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.item()))

            # 日志输出
            if (step + 1) % 100 == 0:
                avg = sum(losses[-100:]) / 100
                print(f"第 {epoch+1}/{epoch_num} 轮 | step {step+1}/{train_steps_per_epoch} | loss={avg:.4f}")

        avg_epoch = sum(losses) / max(1, len(losses))
        print(f"\n========== 第 {epoch+1} 轮结束，平均 loss：{avg_epoch:.4f} ==========\n")

        # 每轮做一个小样例预测
        demo_q = "查一下积分"
        pred = predict_label(demo_q, model, tokenizer, label_list, device)
        print(f"【样例预测】输入：{demo_q} -> 预测标签：{pred}\n")

    # 保存模型
    if save_weight:
        out_dir = os.path.join(os.path.abspath(project_root), "model_bert2bert_sft")
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"模型已保存到：{out_dir}")


if __name__ == "__main__":
    train(".", save_weight=True)
