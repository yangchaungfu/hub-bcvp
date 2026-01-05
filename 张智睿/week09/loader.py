# loader.py
import json
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer


class DataGenerator(Dataset):
    """
    读取 CoNLL/BIO 格式：
      token label
    空行分句

    输出给 BERT：
      input_ids, attention_mask, token_type_ids, labels(piece-level), word_ids(piece-level), token_len
    其中：
      - labels 与 input_ids 同长度，padding 为 -1
      - word_ids: 每个 wordpiece 对应原始 token 的下标；特殊符号/padding 为 -1
      - token_len: 原始 token 数量（用于评估时聚合回 token-level）
    """

    def __init__(self, data_path, config):
        self.config = config
        self.max_length = config["max_length"]

        # schema: label -> id
        with open(config["schema_path"], "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.id2label = {v: k for k, v in self.schema.items()}

        self.o_id = self.schema.get("O", 0)

        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

        self.tokens_list = []   # List[List[str]]
        self.labels_list = []   # List[List[int]]
        self.sentences = []     # List[str]  仅用于显示/对齐

        self._load(data_path)

    def _load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return

        samples = raw.split("\n\n")
        for s in samples:
            lines = [ln for ln in s.split("\n") if ln.strip()]
            tokens = []
            labels = []
            for ln in lines:
                tok, lab = ln.split()
                tokens.append(tok)
                # label -> id（保证 train/test 统一映射）
                labels.append(self.schema[lab])
            self.tokens_list.append(tokens)
            self.labels_list.append(labels)
            self.sentences.append("".join(tokens))

    def __len__(self):
        return len(self.tokens_list)

    def _cont_label_id(self, label_id: int) -> int:
        """
        当一个 token 被 tokenize 成多个 wordpiece 时：
          - 第一个 piece 用原 label
          - 后续 piece 用 I-xxx（如果原是 B-xxx），否则保持不变
        这样可保证训练/CRF 序列连续可学。
        """
        lab = self.id2label.get(label_id, "O")
        if lab == "O":
            return label_id
        if lab.startswith("B-"):
            cont = "I-" + lab[2:]
            return self.schema.get(cont, label_id)
        return label_id  # I-xxx 继续 I-xxx

    def _encode_one(self, tokens, labels):
        # 手工逐 token tokenize，避免 tokenizer 合并/乱拆数据 token（尤其数字）
        wp_tokens = ["[CLS]"]
        wp_labels = [-1]
        wp_word_ids = [-1]

        for i, (tok, lab_id) in enumerate(zip(tokens, labels)):
            pieces = self.tokenizer.tokenize(tok)
            if not pieces:
                pieces = ["[UNK]"]

            for j, p in enumerate(pieces):
                wp_tokens.append(p)
                wp_word_ids.append(i)
                if j == 0:
                    wp_labels.append(lab_id)
                else:
                    wp_labels.append(self._cont_label_id(lab_id))

        wp_tokens.append("[SEP]")
        wp_labels.append(-1)
        wp_word_ids.append(-1)

        # 截断：保证末尾是 [SEP]
        if len(wp_tokens) > self.max_length:
            wp_tokens = wp_tokens[: self.max_length]
            wp_labels = wp_labels[: self.max_length]
            wp_word_ids = wp_word_ids[: self.max_length]
            wp_tokens[-1] = "[SEP]"
            wp_labels[-1] = -1
            wp_word_ids[-1] = -1

        input_ids = self.tokenizer.convert_tokens_to_ids(wp_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # padding 到 max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len
            wp_labels += [-1] * pad_len
            wp_word_ids += [-1] * pad_len

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids),
            torch.LongTensor(wp_labels),
            torch.LongTensor(wp_word_ids),
        )

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        labels = self.labels_list[idx]

        input_ids, attention_mask, token_type_ids, wp_labels, wp_word_ids = self._encode_one(tokens, labels)
        token_len = len(tokens)

        return input_ids, attention_mask, token_type_ids, wp_labels, wp_word_ids, torch.LongTensor([token_len])


def load_data(data_path, config, shuffle=True):
    dataset = DataGenerator(data_path, config)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
