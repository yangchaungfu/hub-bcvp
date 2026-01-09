"""
大语言模型SFT（Supervised Fine-Tuning）微调示例
使用trl框架进行监督式微调
"""
import json

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from trl import SFTTrainer, SFTConfig

# 配置
# 使用本地模型路径，确保可以离线运行
# 首次运行前，请先执行 python download_model.py 下载模型
MODEL_NAME = "./models/Qwen2-0.5B-Instruct"  # 本地模型路径
OUTPUT_DIR = "./output"
MAX_SEQ_LENGTH = 512

# 构造中文问答数据
def create_chinese_dataset():
    """构造简单的中文问答数据集"""
    with open("./corpus.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"数据加载完成，数据类型：{type(data)}")
    # 将数据转换为对话格式（Qwen格式）
    formatted_data = []
    for item in data:
        # 使用Qwen的对话格式
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted_data.append({"text": text})
    return formatted_data

def load_model_and_tokenizer():
    """加载模型和分词器"""
    print(f"正在加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # 使用float32以避免fp16训练时的梯度缩放问题
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # 使用float32以确保训练稳定性
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main():
    """主训练函数"""
    print("=" * 50)
    print("开始SFT微调训练")
    print("=" * 50)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 创建数据集
    print("\n正在构造训练数据...")
    train_data = create_chinese_dataset()
    dataset = Dataset.from_list(train_data)
    
    print(f"训练样本数量: {len(dataset)}")
    
    # 训练参数 - 使用SFTConfig（新版本trl推荐）
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # batch size适合普通电脑
        gradient_accumulation_steps=4,  # 通过梯度累积增大有效batch size
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=False,  # 禁用fp16以避免梯度缩放问题
        bf16=False,  # 禁用bf16
        remove_unused_columns=False,
        report_to=None,  # 不使用wandb等
        dataloader_pin_memory=False,  # Windows上可能需要设置为False
        max_length=MAX_SEQ_LENGTH,  # 最大序列长度（注意：新版本使用max_length而不是max_seq_length）
        dataset_text_field="text",  # 指定数据集中文本字段的名称
    )
    
    # 创建SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,  # 新版本使用processing_class而不是tokenizer
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存模型
    print(f"\n训练完成，正在保存模型到 {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("=" * 50)
    print("训练完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

