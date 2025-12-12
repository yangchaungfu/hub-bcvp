## 20251207-week08-第八周作业

### 作业内容和要求

#### 基于已有的项目，使用三元组损失完成文本匹配模型训练

- 当前项目是一个文本匹配系统，用于判断两个句子之间的相似度
- 原始实现使用余弦嵌入损失（Cosine Embedding Loss）作为训练目标
- 现在需要将其修改为使用三元组损失（Triplet Loss），以提高模型的表示能力和匹配性能

### 修改目标

1. 将损失函数从余弦嵌入损失替换为三元组损失
2. 更新模型的前向传播方法，支持三元组输入（anchor, positive, negative）
3. 修改数据加载器，生成三元组样本用于训练
4. 更新训练循环，适配三元组损失的计算方式
5. 确保测试阶段的代码能够正常工作

## 实现方案

### 1. 模型修改: [model.py](model.py)

#### 1.1 损失函数替换

将原始的余弦嵌入损失替换为自定义的余弦三元组损失：

```python
# 原始代码
self.loss = nn.CosineEmbeddingLoss()

# 修改后代码
self.loss = self.cosine_triplet_loss
```

#### 1.2 前向传播方法更新

更新`forward`方法，支持三种输入模式：
- 单个句子输入：返回句子向量（用于测试阶段）
- 两个句子输入：计算余弦损失（保留向后兼容）
- 三个句子输入：计算三元组损失（用于训练阶段）

```python
def forward(self, sentence1, sentence2=None, sentence3=None):
    # 如果只有一个句子输入，说明是测试阶段，返回句子向量
    if sentence2 is None:
        return self.sentence_encoder(sentence1)
    # 如果有sentence3，说明是训练阶段，计算triplet loss
    elif sentence3 is not None:
        vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
        vector2 = self.sentence_encoder(sentence2)
        vector3 = self.sentence_encoder(sentence3)
        loss = self.loss(vector1, vector2, vector3)
        return loss
    # 如果只有两个句子输入，计算cosine loss（保留原有接口兼容）
    else:
        vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
        vector2 = self.sentence_encoder(sentence2)
        loss = self.cosine_distance(vector1, vector2)
        return loss
```

#### 1.3 三元组损失函数实现

实现改进的余弦三元组损失函数，避免出现nan值：

```python
def cosine_triplet_loss(self, a, p, n, margin=None):
    # 计算a和p、a和n的余弦距离
    ap = self.cosine_distance(a,p)
    an = self.cosine_distance(a,n)
    # 如果没有设置margin，则设置margin为0.1
    if margin is None:
        margin = 0.1
    else:
        margin = margin.squeeze()
    # 计算triplet loss: max(ap - an + margin, 0)
    diff = ap - an + margin
    # 确保不会出现空tensor导致nan
    loss = torch.mean(torch.max(diff, torch.zeros_like(diff)))
    return loss
```

### 2. 训练循环修改 [main.py](main.py)

更新训练循环，适配三元组损失的计算方式：

```python
# 原始代码
input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
loss = model(input_id1, input_id2, labels)

# 修改后代码
input_id1, input_id2, input_id3 = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
loss = model(input_id1, input_id2, input_id3)
```

### 3. 数据加载器修改 [loader.py](loader.py)

#### 3.1 随机训练样本生成

更新`random_train_sample`方法，生成三元组样本：

```python
# 原始代码（二分类样本）
def random_train_sample(self):
    standard_question_index = list(self.knwb.keys())
    #随机正样本
    if random.random() <= self.config["positive_sample_rate"]:
        p = random.choice(standard_question_index)
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
            return [s1, s2, torch.LongTensor([1])]
    #随机负样本
    else:
        p, n = random.sample(standard_question_index, 2)
        s1 = random.choice(self.knwb[p])
        s2 = random.choice(self.knwb[n])
        return [s1, s2, torch.LongTensor([-1])]

# 修改后代码（三元组样本）
def random_train_sample(self):
    standard_question_index = list(self.knwb.keys())
    p,q = random.sample(standard_question_index, 2)
    #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
    if  len(self.knwb[p]) < 2:
        return self.random_train_sample()
    else:
        s1, s2 = random.sample(self.knwb[p], 2)
        s3 = random.choice(self.knwb[q])
        return [s1, s2, s3]
```

## 效果验证

### 1. 训练损失

修改后模型能够正常训练，损失值逐步降低：

```
2025-12-12 01:19:19,796 - __main__ - INFO - epoch average loss: 0.006869
```

### 2. 测试准确率

模型在测试集上的表现：

```
2025-12-12 01:19:19,880 - __main__ - INFO - 预测集合条目总量：464
2025-12-12 01:19:19,880 - __main__ - INFO - 预测正确条目：419，预测错误条目：45
2025-12-12 01:19:19,880 - __main__ - INFO - 预测准确率：0.903017
```

### 3. 问题修复

- 解决了nan loss问题：通过修改三元组损失函数的实现，确保不会出现空tensor导致的nan值
- 确保了测试阶段的兼容性：修改后的模型能够在测试阶段正确处理单个句子输入

## 总结

本次修改成功将文本匹配系统的损失函数从余弦嵌入损失替换为三元组损失，主要完成了以下工作：

1. **模型层面**：更新了损失函数和前向传播方法，支持三元组输入和计算
2. **训练层面**：修改了训练循环，适配三元组损失的计算方式
3. **数据层面**：更新了数据加载器，生成三元组样本用于训练
4. **兼容性**：确保了测试阶段的代码能够正常工作

修改后的模型能够稳定训练，损失值逐步降低，测试准确率达到约90.3%，证明了三元组损失的有效性。

## 文件修改列表

| 文件路径                    | 修改内容 |
|-------------------------|---------|
| [model.py](model.py)    | 替换损失函数、更新前向传播方法、修复三元组损失实现 |
| [main.py](main.py)      | 更新训练循环，适配三元组输入 |
| [loader.py ](loader.py) | 修改数据加载器，生成三元组样本 |