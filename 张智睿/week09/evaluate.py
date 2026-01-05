# evaluate.py
import json
import torch


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger

        with open(config["schema_path"], "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.id2label = {v: k for k, v in self.schema.items()}
        self.o_id = self.schema.get("O", 0)

        from loader import load_data
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

        self.stats = None
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "LOCATION": {"correct": 0, "pred": 0, "true": 0},
            "ORGANIZATION": {"correct": 0, "pred": 0, "true": 0},
            "PERSON": {"correct": 0, "pred": 0, "true": 0},
            "TIME": {"correct": 0, "pred": 0, "true": 0},
        }

    @torch.no_grad()
    def eval(self, epoch):
        self.reset_stats()
        self.model.eval()

        dataset = self.valid_data.dataset

        for idx, batch in enumerate(self.valid_data):
            input_ids, attention_mask, token_type_ids, labels, word_ids, token_len = batch
            token_len = token_len.squeeze(1)  # [B]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                labels = labels.cuda()
                word_ids = word_ids.cuda()

            pred_piece = self.model(input_ids, attention_mask, token_type_ids)  # [B, T]（-1 表示无效）

            # 取原句 tokens（与 batch 对齐）
            start = idx * self.config["batch_size"]
            end = start + input_ids.size(0)
            tokens_batch = dataset.tokens_list[start:end]

            self.write_stats(tokens_batch, labels, pred_piece, word_ids, token_len)

        self.show_stats(epoch)

    def _collapse_to_token_level(self, piece_ids, piece_word_ids, token_len):
        """
        piece_ids:   [T]  预测或真值（piece-level）
        piece_word_ids: [T] 每个 piece 属于哪个原 token，下标；特殊/pad 为 -1
        返回：token-level label_ids，长度 token_len
        规则：取每个 token 的“第一个 piece”的标签
        """
        out = [self.o_id] * token_len
        seen = set()
        for lab, wid in zip(piece_ids, piece_word_ids):
            wid = int(wid)
            if wid < 0 or wid >= token_len:
                continue
            if wid in seen:
                continue
            lab = int(lab)
            out[wid] = lab
            seen.add(wid)
        return out

    def _extract_entities(self, tokens, label_ids):
        """
        标准 BIO 扫描抽取实体，支持单字实体。
        返回 dict: {TYPE: [entity_text,...]}
        """
        entities = {"LOCATION": [], "ORGANIZATION": [], "PERSON": [], "TIME": []}
        n = len(tokens)
        i = 0
        while i < n:
            lab = self.id2label.get(label_ids[i], "O")
            if lab.startswith("B-"):
                etype = lab[2:]
                start = i
                i += 1
                while i < n and self.id2label.get(label_ids[i], "O") == f"I-{etype}":
                    i += 1
                if etype in entities:
                    entities[etype].append("".join(tokens[start:i]))
            else:
                i += 1
        return entities

    def write_stats(self, tokens_batch, true_piece, pred_piece, word_ids, token_len):
        true_piece = true_piece.detach().cpu().tolist()
        pred_piece = pred_piece.detach().cpu().tolist()
        word_ids = word_ids.detach().cpu().tolist()
        token_len = token_len.detach().cpu().tolist()

        for tokens, t_p, p_p, w_p, tl in zip(tokens_batch, true_piece, pred_piece, word_ids, token_len):
            tl = int(tl)

            true_tok = self._collapse_to_token_level(t_p, w_p, tl)
            pred_tok = self._collapse_to_token_level(p_p, w_p, tl)

            true_entities = self._extract_entities(tokens, true_tok)
            pred_entities = self._extract_entities(tokens, pred_tok)

            for k in self.stats.keys():
                t_list = true_entities[k]
                p_list = pred_entities[k]
                self.stats[k]["true"] += len(t_list)
                self.stats[k]["pred"] += len(p_list)

                # 完全匹配计正确（保持你原项目的统计口径）
                t_set = set(t_list)
                for ent in p_list:
                    if ent in t_set:
                        self.stats[k]["correct"] += 1

    def show_stats(self, epoch):
        self.logger.info("=========%s=========" % epoch)

        f1s = []
        correct_sum = pred_sum = true_sum = 0

        for k, v in self.stats.items():
            c, p, t = v["correct"], v["pred"], v["true"]
            correct_sum += c
            pred_sum += p
            true_sum += t

            precision = c / p if p else 0.0
            recall = c / t if t else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            f1s.append(f1)

            self.logger.info("%s:\tP=%.4f\tR=%.4f\tF1=%.4f\t(c=%d,p=%d,t=%d)" % (k, precision, recall, f1, c, p, t))

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        micro_p = correct_sum / pred_sum if pred_sum else 0.0
        micro_r = correct_sum / true_sum if true_sum else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

        self.logger.info("Macro-F1: %.4f" % macro_f1)
        self.logger.info("Micro-F1: %.4f (P=%.4f, R=%.4f)" % (micro_f1, micro_p, micro_r))
