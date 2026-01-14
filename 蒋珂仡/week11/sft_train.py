"""
大语言模型SFT（Supervised Fine-Tuning）微调示例
使用trl框架进行监督式微调
"""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
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
    """
    构造 Seq2Seq 任务数据集：中译英翻译
    作业目标：让模型学会把中文翻译成英文
    """
    data = [
        # 我们构造一些简单的翻译对，让模型过拟合这些模式
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "你好。",
            "output": "Hello."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "很高兴见到你。",
            "output": "Nice to meet you."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "今天天气真好。",
            "output": "The weather is really good today."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "深度学习很有趣。",
            "output": "Deep learning is very interesting."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "苹果是一种水果。",
            "output": "Apple is a kind of fruit."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "我想喝杯咖啡。",
            "output": "I want to have a cup of coffee."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "目前的时间是几点？",
            "output": "What time is it now?"
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "这个任务很简单。",
            "output": "This task is very simple."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "谢谢你的帮助。",
            "output": "Thank you for your help."
        },
        {
            "instruction": "将下面的中文翻译成英文。",
            "input": "明天会下雨吗？",
            "output": "Will it rain tomorrow?"
        }
    ]

    # 将数据转换为对话格式（Qwen格式）
    formatted_data = []
    for item in data:
        # 这里的关键是把 instruction 和 input 拼在用户输入里
        user_input = f"{item['instruction']}\n{item['input']}"

        # 构造 Seq2Seq 的 Prompt
        # 输入序列 (Source): 用户指令 + 中文原句
        # 输出序列 (Target): 英文翻译
        text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
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

