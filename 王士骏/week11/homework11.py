import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# 配置参数
model_name = r"D:\W11\Python\第06周 语言模型\bert-base-chinese"
max_seq_len = 128
batch_size = 64
lr = 1e-4
epochs = 150

# 初始化tokenizer和BertModel
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)


class BertSeq2Seq(nn.Module):
    def __init__(self, bert_model, vocab_size, hidden_dim=768):
        super().__init__()
        self.bert = bert_model
        # 自定义分类头：将BERT的hidden_state映射到词表维度
        self.cls_head = nn.Linear(hidden_dim, vocab_size)
        # 初始化分类头权重（可选，提升收敛速度）
        nn.init.normal_(self.cls_head.weight, std=0.02)
        nn.init.constant_(self.cls_head.bias, 0)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]  # 获取序列长度
        # 生成下三角矩阵（1=允许关注，0=禁止关注后文）
        autoregressive_mask = torch.tril(torch.ones(seq_len, seq_len))
        # 结合padding_mask（attention_mask）和自回归掩码
        # attention_mask形状：[batch, seq_len] → 扩展为[batch, seq_len, seq_len]
        padding_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, seq_len)
        # 最终注意力掩码：padding_mask * 自回归掩码（双重过滤）
        final_attention_mask = padding_mask * autoregressive_mask

        # BERT前向传播：输出last_hidden_state [batch, seq_len, 768]
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,encoder_attention_mask=final_attention_mask)
        last_hidden_state,_ = bert_output  # 核心输出
        # 映射到词表维度 [batch, seq_len, vocab_size]
        logits = self.cls_head(last_hidden_state)
        return logits

# 初始化完整模型
vocab_size = tokenizer.vocab_size
model = BertSeq2Seq(bert_model, vocab_size)


class QADataSet(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_len):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_token_id = tokenizer.mask_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]

        # 步骤1：拼接序列 [CLS] 问题 [SEP] 回答 [SEP]
        # 分别编码问题和回答（避免截断时破坏回答结构）
        q_encoding = self.tokenizer(
            question, truncation=True, max_length=self.max_len // 2, add_special_tokens=False
        )
        a_encoding = self.tokenizer(
            answer, truncation=True, max_length=self.max_len // 2 - 2, add_special_tokens=False
        )

        # 构造完整input_ids
        input_ids = [self.tokenizer.cls_token_id] + q_encoding["input_ids"] + [self.sep_token_id] + \
                    a_encoding["input_ids"] + [self.sep_token_id]

        # 步骤2：构造Mask后的input_ids（回答部分替换为[MASK]）
        masked_input_ids = input_ids.copy()
        # 找到回答部分的起始/结束位置（sep后到最后一个sep前）
        sep_positions = [i for i, id in enumerate(input_ids) if id == self.sep_token_id]
        ans_start = sep_positions[0] + 1
        ans_end = sep_positions[1] if len(sep_positions) >= 2 else len(input_ids) - 1
        # 替换回答部分为[MASK]
        for i in range(ans_start, ans_end):
            masked_input_ids[i] = self.mask_token_id

        # 步骤3：构造attention_mask（核心！1=有效，0=padding）
        attention_mask = [1] * len(masked_input_ids)

        # 步骤4：padding到max_len
        padding_len = self.max_len - len(masked_input_ids)
        if padding_len > 0:
            masked_input_ids += [self.pad_token_id] * padding_len
            attention_mask += [0] * padding_len  # padding部分attention_mask设为0
        else:
            masked_input_ids = masked_input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]

        # 步骤5：构造labels（仅回答部分保留真实token，其余为-100）
        labels = [-100] * self.max_len  # CrossEntropy忽略-100
        # 真实回答的token_id（原input_ids中的回答部分）
        true_ans_ids = input_ids[ans_start:ans_end]
        # 对齐labels的位置（仅填充回答部分）
        if ans_start < self.max_len:
            labels[ans_start: min(ans_start + len(true_ans_ids), self.max_len)] = true_ans_ids[
                                                                                  :self.max_len - ans_start]

        # 转换为tensor
        masked_input_ids = torch.tensor(masked_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# 示例QA数据集（替换为自己的SFT数据）

def load_data(file_path):
    result_list = []
    len_anw = 50
    with open(file_path, 'r', encoding='utf8') as f:
        lines = [line.strip() for line in f if line.strip()]
        for line_num, line in enumerate(lines, 1):
            json_line = json.loads(line)
            if isinstance(json_line, dict):
                title = json_line.get('title', '')
                content = json_line.get('content', '')[:len_anw]
                result_list.append((title, content))
    return result_list

qa_pairs = load_data('train_tag_news.json')

# qa_pairs = [
#     ("什么是人工智能？", "人工智能是模拟人类智能的技术，涵盖机器学习、自然语言处理等领域。"),
#     ("BERT的核心特点是什么？", "BERT是基于Transformer的Encoder模型，采用双向注意力机制，擅长上下文理解。"),
#     ("如何训练Seq2Seq模型？", "Seq2Seq模型通常包含Encoder和Decoder，通过监督学习最小化输出序列的交叉熵损失。"),
#     ("PyTorch中view和reshape的区别？", "view基于浅视图不复制数据，要求张量连续；reshape更通用，可能复制数据。"),
#     ("什么是SFT训练？", "SFT是监督微调，用标注数据调整预训练模型，适配特定任务。")
# ]

# 构建DataLoader
dataset = QADataSet(qa_pairs, tokenizer, max_seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 优化器与损失函数
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略labels=-100的位置

# 训练循环
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch in dataloader:
        # 数据移到设备
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # 前向传播
        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [batch, seq_len, vocab_size]

        # 计算损失：仅关注labels≠-100的位置（回答部分）
        # 调整维度：logits -> [batch*seq_len, vocab_size], labels -> [batch*seq_len]
        loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "bert_seq2seq_sft.pth")
tokenizer.save_pretrained("bert_seq2seq_tokenizer")


def generate_answer(question, model, tokenizer, max_len=128, gen_mode="step"):
    model.eval()
    answer_max_len = 32
    input_text = f"{question}"
    q_encoding = tokenizer(
        input_text, truncation=True, max_length=max_len // 2, add_special_tokens=False
    )
    input_ids = [tokenizer.cls_token_id] + q_encoding["input_ids"] + [tokenizer.sep_token_id] + \
                [tokenizer.mask_token_id] * answer_max_len + [tokenizer.sep_token_id]
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
    attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)

    sep_pos = (input_ids[0] == tokenizer.sep_token_id).nonzero()[0].item()
    ans_start = sep_pos + 1
    ans_end = ans_start + answer_max_len

    with torch.no_grad():
        if gen_mode == "batch":
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            mask_logits = logits[0, ans_start:ans_end, :]
            pred_token_ids = torch.argmax(mask_logits, dim=-1)
            pred_tokens = tokenizer.convert_ids_to_tokens(pred_token_ids.cpu().numpy())
            answer = "".join([t for t in pred_tokens if t not in [tokenizer.pad_token, tokenizer.mask_token]])

        elif gen_mode == "step":
            current_input_ids = input_ids.clone()
            for i in range(ans_start, min(ans_end, max_len - 1)):
                # ========== 新增：推理阶段构造自回归掩码 ==========
                current_attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in
                                          current_input_ids[0].cpu().numpy()]
                current_attention_mask = torch.tensor([current_attention_mask], dtype=torch.long)
                # 生成当前序列长度的下三角掩码
                seq_len = current_input_ids.shape[1]
                autoregressive_mask = torch.tril(torch.ones(seq_len, seq_len))
                padding_mask = current_attention_mask.unsqueeze(1).expand(-1, seq_len, seq_len)
                final_attention_mask = padding_mask * autoregressive_mask
                # ========== 自回归掩码构造结束 ==========

                # 前向传播：传入final_attention_mask（修改模型forward后，此处只需传基础attention_mask）
                # 注：若模型forward已集成自回归掩码，此处只需传current_attention_mask即可
                logits = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                pred_token_id = torch.argmax(logits[0, i, :], dim=-1)
                current_input_ids[0, i] = pred_token_id
                if pred_token_id == tokenizer.sep_token_id:
                    break
            pred_token_ids = current_input_ids[0, ans_start:i].cpu().numpy()
            answer = tokenizer.decode(pred_token_ids, skip_special_tokens=True)

    return answer


# 测试推理
test_question = "什么是人工智能？"
answer_batch = generate_answer(test_question, model, tokenizer, gen_mode="batch")
answer_step = generate_answer(test_question, model, tokenizer, gen_mode="step")
print(f"问题：{test_question}")
print(f"批量生成回答：{answer_batch}")
print(f"逐token生成回答：{answer_step}")

test_question = "世界有哪些奇葩建筑？"
answer_batch = generate_answer(test_question, model, tokenizer, gen_mode="batch")
answer_step = generate_answer(test_question, model, tokenizer, gen_mode="step")
print(f"问题：{test_question}")
print(f"批量生成回答：{answer_batch}")
print(f"逐token生成回答：{answer_step}")
