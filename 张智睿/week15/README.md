# 张智睿-第15周作业，BPE 分词器

## 两个可直接运行程序
1) `build_tokenizer.py`
- 读取 `data/长安乱.txt`
- 训练一个 **50 词规模** 的 BPE 词表
- 输出 `bpe_project/tokenizer.json`（包含词表与 token-id 对应关系、merges）

2) `demo_encode_decode.py`
- 加载 `tokenizer.json`
- 演示 `encode(text)->token_ids` 与 `decode(token_ids)->text`

## 项目结构
```
bpe_project/
  bpe_trainer.py
  tokenizer.py
  build_tokenizer.py
  demo_encode_decode.py
  tokenizer.json          # build 后生成
data/
  长安乱.txt
```
