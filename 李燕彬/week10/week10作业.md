## 20251221-week10-第十周作业

### 1. 作业内容和要求

#### 基于已有的项目，通过BERT实现自回归语言模型训练

- 本项目基于 BERT 预训练模型实现中文语言模型训练
- 支持 **Mask Attention（掩码注意力）** 机制，能够根据给定文本片段生成后续内容。
- 项目使用 PyTorch 框架，结合 Hugging Face Transformers 库，支持 GPU 加速训练。

### 2. 文件结构

```
bert语言模型生成文本/
├── config.py           # 配置文件，包含模型参数和训练超参数
├── model.py            # BERT语言模型实现，包含Mask Attention机制
├── main.py             # 主程序入口，负责模型训练流程控制
├── loader.py           # 数据加载模块，处理语料库和词汇表
├── evaluate.py         # 评估模块，实现文本生成功能
├── corpus.txt          # 训练语料库（约4MB中文文本）
└── vocab.txt           # 词汇表文件
```

### 3. 核心模块说明

#### 3.1 config.py - 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `corpus_path` | `"corpus.txt"` | 语料库文件路径 |
| `vocab_path` | `"vocab.txt"` | 词汇表文件路径 |
| `hidden_size` | `768` | BERT模型隐藏层维度（bert-base-chinese） |
| `learning_rate` | `1e-3` | 学习率 |
| `optimizer` | `"adam"` | 优化器类型 |
| `epoch_num` | `20` | 训练轮数 |
| `batch_size` | `64` | 每次训练的样本数量 |
| `train_sample` | `50000` | 每轮训练的总样本数 |
| `window_size` | `10` | 输入窗口大小 |
| `text_length` | `50` | 生成文本的最大长度 |
| `bert_path` | `bert-base-chinese路径` | BERT预训练模型路径 |

#### 3.2 model.py - 模型实现

##### 3.2.1 Mask Attention 机制

本项目实现了完整的 Mask Attention 机制，包含以下关键组件：

**（1）因果掩码（Causal Mask）**

```python
def subsequent_mask(size):
    """
    生成因果掩码，防止模型在预测时看到未来的token
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

**原理**：使用上三角矩阵，将当前位置之后的所有位置标记为被遮蔽，确保位置 `i` 只能关注位置 `0` 到 `i` 的内容。

**（2）填充掩码（Padding Mask）**

```python
def pad_mask(inputs, pad_id=0):
    """
    生成填充掩码，忽略输入序列中的padding token
    """
    input_mask = (inputs != pad_id).unsqueeze(1).unsqueeze(2)
    return input_mask
```

**原理**：标记输入中非 padding 的位置，用于处理变长序列。

**（3）组合掩码**

```python
def forward(self, x, y=None):
    # 生成填充掩码
    input_mask = (x != self.pad_id).unsqueeze(1)
    
    # 生成因果掩码
    seq_len = x.size(1)
    causal_mask = subsequent_mask(seq_len).to(x.device)
    
    # 组合掩码
    combined_mask = input_mask & causal_mask
    attention_mask = combined_mask.squeeze(1)
    
    # 传入BERT模型
    x, _ = self.bert(x, attention_mask=attention_mask)
```

**流程**：
1. 生成填充掩码，标记非 padding 位置
2. 生成因果掩码，防止 attending 到未来位置
3. 组合两种掩码，同时满足非 padding 且不是未来位置
4. 将组合掩码传入 BERT 模型

##### 3.2.2 模型架构

```
输入序列 (batch_size, seq_len)
    ↓
BERT Encoder (bert-base-chinese)
    ↓
输出: (batch_size, seq_len, 768)
    ↓
Linear Classification Layer (768 → vocab_size)
    ↓
Softmax / CrossEntropyLoss
```

#### 3.3 loader.py - 数据加载

**主要函数**：

| 函数 | 功能 |
|------|------|
| `load_vocab(path)` | 加载词汇表，返回字典 `{字: 索引}` |
| `load_corpus(path)` | 加载语料库，返回拼接的文本字符串 |
| `build_sample(vocab, window_size, corpus)` | 随机生成一个训练样本 |
| `build_dataset(sample_length, vocab, window_size, corpus)` | 构建数据集 |

**训练样本构造**：
- 从语料库中随机截取长度为 `window_size` 的窗口
- 输入：窗口前 `window_size` 个字
- 输出：窗口后 `window_size` 个字（错开一位）

#### 3.4 main.py - 主程序

**训练流程**：

```
初始化模型
    ↓
加载BERT预训练权重
    ↓
for epoch in range(epoch_num):
    for batch in range(train_sample / batch_size):
        构建训练样本
        前向传播（包含Mask Attention）
        计算loss
        反向传播
        更新权重
        输出loss（每255个batch）
    输出平均loss
    生成示例文本
```

**关键特性**：
- 支持 CUDA GPU 加速
- 实时输出训练 loss
- 每轮结束后生成示例文本
- 支持模型权重保存

#### 3.5 evaluate.py - 评估与生成

**文本生成算法**：

```python
def generate_sentence(openings, model, vocab, window_size):
    # 循环生成直到遇到换行符或达到最大长度
    while pred_char != "\n" and len(openings) <= text_length:
        # 取最后window_size个字作为输入
        x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
        # 模型预测下一个字
        y = model(x)[0][-1]
        # 采样策略：90%贪心，10%随机采样
        index = sampling_strategy(y)
    return openings
```

**采样策略**：
- **贪心策略**：选择概率最高的字（90%概率）
- **随机采样**：按概率分布随机选择字（10%概率）

### 4. 环境要求

```
Python 3.7+
PyTorch 1.8+
Transformers 4.0+
NumPy
```

### 5. 运行说明

#### 5.1 准备预训练模型

下载 BERT 中文预训练模型到指定路径：
```python
bert_path = r"E:\BaiduNetdiskDownload\bert-base-chinese"
```

#### 5.2 运行训练

```bash
cd bert语言模型生成文本
python main.py
```

#### 5.3 预期输出

```
是否使用cuda：True
文本词表模型加载完毕，开始训练
Epoch 1, Batch 0, Loss: 8.4222
Epoch 1, Batch 255, Loss: 5.2134
Epoch 1, Batch 510, Loss: 4.8921
...
=========
第1轮平均loss:4.5231
让他在半年之前，就不能做出...（生成的文本）
李慕站在山路上，深深的呼吸...（生成的文本）
```

### 6. Mask Attention 机制详解

#### 6.1 为什么需要 Mask Attention？

1. **防止信息泄露**：在自回归语言模型中，预测位置 `t` 时不应看到位置 `t+1` 及之后的token

2. **处理变长序列**：通过 padding mask 忽略 padding token，保证不同长度序列的正常处理

3. **提高训练效率**：掩码机制使得模型只能利用已生成的信息进行预测

#### 6.2 掩码矩阵示例

假设序列长度为 5：

**因果掩码**：
```
[[ True, False, False, False, False],
 [ True,  True, False, False, False],
 [ True,  True,  True, False, False],
 [ True,  True,  True,  True, False],
 [ True,  True,  True,  True,  True]]
```

**填充掩码**（假设第2个位置是padding）：
```
[[ True, False,  True,  True,  True],
 [ True, False,  True,  True,  True],
 ...]
```

**组合掩码**：
```
[[ True, False, False, False, False],
 [ True, False, False, False, False],
 [ True,  True,  True, False, False],
 [ True,  True,  True,  True, False],
 [ True,  True,  True,  True,  True]]
```

### 7. 训练技巧

1. **学习率调度**：可使用 warmup 策略，初始学习率较小，逐渐增加到目标值

2. **梯度裁剪**：防止梯度爆炸，限制梯度范数在合理范围内

3. **早停策略**：监控验证集 loss，连续多轮不下降时停止训练

4. **模型保存**：定期保存模型权重，便于恢复训练

### 8. 常见问题

#### Q1: CUDA 不可用？
确保已安装支持 CUDA 的 PyTorch 版本：
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
```

#### Q2: 内存不足？
减小 `batch_size` 或 `train_sample` 的值

#### Q3: 训练 loss 不下降？
1. 检查学习率是否合适
2. 确认 BERT 模型路径正确
3. 增加训练轮数

### 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch for CUDA](https://pytorch.org/get-started/previous-versions/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/v5.0.0rc1/zh/index)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
