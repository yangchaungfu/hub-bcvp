# model.py
import torch
import torch.nn as nn

from torchcrf import CRF
from transformers import BertModel


class TorchModel(nn.Module):
    """
    兼容原项目命名 TorchModel，但内部改为 BERT。
    forward:
      - 训练：传 labels -> 返回 loss
      - 推理：不传 labels -> 返回 pred_ids (Tensor: [B, T]，padding/special 为 -1)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.class_num = config["class_num"]
        self.use_crf = bool(config.get("use_crf", False))

        self.bert = BertModel.from_pretrained(config["bert_path"])
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self.classifier = nn.Linear(hidden, self.class_num)

        if self.use_crf:
            self.crf = CRF(self.class_num, batch_first=True)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq = self.dropout(out.last_hidden_state)  # [B, T, H]
        logits = self.classifier(seq)              # [B, T, C]

        if labels is not None:
            if self.use_crf:
                # CRF 不建议包含 [CLS]/[SEP]，我们直接去掉首尾位置
                emissions = logits[:, 1:-1, :]
                tags = labels[:, 1:-1]
                mask = (attention_mask[:, 1:-1] > 0) & tags.gt(-1)

                # torchcrf 返回 log_likelihood，训练用负号当 loss
                llh = self.crf(emissions, tags, mask=mask, reduction="mean")
                return -llh
            else:
                return self.ce_loss(logits.reshape(-1, self.class_num), labels.reshape(-1))

        # 推理：输出 token id 序列（对齐 input_ids 长度，special/pad 填 -1）
        if self.use_crf:
            emissions = logits[:, 1:-1, :]
            mask = attention_mask[:, 1:-1] > 0
            decoded = self.crf.decode(emissions, mask=mask)  # List[List[int]]

            B, T = input_ids.shape
            pred = torch.full((B, T), -1, dtype=torch.long, device=input_ids.device)
            for b in range(B):
                seq_ids = decoded[b]
                # 放回到 [1:1+len]（避开 CLS）
                end = min(1 + len(seq_ids), T - 1)  # 预留 SEP
                pred[b, 1:end] = torch.tensor(seq_ids[: end - 1], device=input_ids.device, dtype=torch.long)
            return pred
        else:
            pred = logits.argmax(dim=-1)
            # 把 padding 位也标成 -1，方便后续评估/对齐
            if attention_mask is not None:
                pred = pred.masked_fill(attention_mask == 0, -1)
            return pred


def choose_optimizer(config, model):
    lr = config["learning_rate"]
    name = str(config.get("optimizer", "adamw")).lower()

    if name in ("adamw", "adam"):
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
