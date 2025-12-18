# self_check.py
import os
import random
import json
import torch

from config import Config

def read_conll(path):
    sents = []
    cur = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            tok, lab = line.split()
            cur.append((tok, lab))
    if cur:
        sents.append(cur)
    return sents

def main():
    # 1) 基础文件存在性检查（不改文件名的前提下）
    must = [
        Config["schema_path"],
        Config["train_data_path"],
        Config["valid_data_path"],
        "ner_data/test",              # 你要求重点确认测试集数字对齐
    ]
    for p in must:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    schema = json.load(open(Config["schema_path"], encoding="utf-8"))
    assert "O" in schema, "schema.json missing 'O'"

    # 2) 数字 token 检查：确认 test 中数字都是单字符，并验证 tokenizer 对单字符数字不会拆分成多 piece
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(Config["bert_path"])

    test_sents = read_conll("ner_data/test")
    digit_tokens = []
    for sent in test_sents:
        for t, _ in sent:
            if t.isdigit():
                digit_tokens.append(t)

    assert digit_tokens, "No digit tokens found in test set (unexpected given your report)."
    assert all(len(t) == 1 for t in digit_tokens), "Found multi-digit tokens in test; please re-check dataset."

    # 验证单个数字 tokenize 结果为单 piece（不会被拆成多 piece）
    bad = []
    for d in random.sample(digit_tokens, min(50, len(digit_tokens))):
        pieces = tok.tokenize(d)
        if len(pieces) != 1:
            bad.append((d, pieces))
    assert not bad, f"Digit token got split unexpectedly: {bad[:5]}"

    print("[PASS] Digit tokens are single-char and tokenizer keeps them as single piece.")

    # 3) loader 对齐检查：CLS/SEP 不应有 label；token/label 映射不乱
    from loader import DataGenerator
    dg = DataGenerator("ner_data/test", Config)

    # 抽样检查若干条
    for idx in random.sample(range(len(dg)), min(10, len(dg))):
        input_ids, attn, ttype, labels, word_ids, token_len = dg[idx]
        # CLS/SEP label 必为 -1
        assert labels[0].item() == -1, "CLS got a label (should be -1)"
        # 找到第一个 SEP（attn==1 且 token==[SEP] 的位置）
        sep_id = tok.convert_tokens_to_ids(["[SEP]"])[0]
        sep_pos = (input_ids == sep_id).nonzero(as_tuple=True)[0][0].item()
        assert labels[sep_pos].item() == -1, "SEP got a label (should be -1)"

        # token-level 长度一致
        tl = token_len.item()
        assert tl == len(dg.tokens_list[idx]), "token_len mismatch"

        # 每个 token 的第一个 piece 对应同一个 word_id
        # 并且 word_id=-1 的位置（CLS/SEP/PAD）不参与 token 对齐
        w = word_ids.tolist()
        for p, wid in enumerate(w):
            if wid == -1:
                continue
            if wid >= tl:
                raise AssertionError("word_id out of range")

    print("[PASS] CLS/SEP have no labels; word_ids/token_len look consistent.")

    # 4) 端到端：跑一个 batch 的 loss + 跑一次 evaluator
    from loader import load_data
    from model import TorchModel, choose_optimizer
    from evaluate import Evaluator
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("self_check")

    data = load_data(Config["train_data_path"], {**Config, "batch_size": 2}, shuffle=True)
    model = TorchModel(Config)
    if torch.cuda.is_available():
        model.cuda()

    batch = next(iter(data))
    input_ids, attn, ttype, labels, word_ids, token_len = batch
    if torch.cuda.is_available():
        input_ids, attn, ttype, labels = input_ids.cuda(), attn.cuda(), ttype.cuda(), labels.cuda()

    loss = model(input_ids, attn, ttype, labels)
    assert torch.isfinite(loss).item(), "loss is not finite"
    print("[PASS] One forward loss computed:", float(loss))

    evaluator = Evaluator(Config, model, logger)
    evaluator.eval(0)
    print("[PASS] Evaluator ran.")

if __name__ == "__main__":
    main()