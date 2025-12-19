# check_model.py
import os
from transformers import BertTokenizer, BertModel

def check_model_directory(model_path):
    print(f"检查模型目录: {model_path}")
    
    if not os.path.exists(model_path):
        print("✗ 目录不存在!")
        return False
    
    # 列出目录内容
    print("目录内容:")
    files = os.listdir(model_path)
    for file in files:
        print(f"  - {file}")
    
    # 检查必要文件
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 缺失")
            missing_files.append(file)
    
    # 检查快速tokenizer所需文件
    fast_tokenizer_files = ['tokenizer.json', 'tokenizer_config.json']
    has_fast_tokenizer = any(f in files for f in fast_tokenizer_files)
    
    if has_fast_tokenizer:
        print("✓ 检测到快速tokenizer文件")
    else:
        print("✗ 缺少快速tokenizer文件，只有慢速tokenizer")
    
    return len(missing_files) == 0

def test_tokenizer(model_path):
    print("\n测试tokenizer加载...")
    
    # 测试快速tokenizer
    try:
        print("尝试加载快速tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_path, use_fast=True)
        print("✓ 快速tokenizer加载成功")
        
        # 测试word_ids方法
        test_text = "这是一个测试"
        encoding = tokenizer(test_text, return_tensors='pt')
        try:
            word_ids = encoding.word_ids()
            print(f"✓ word_ids()方法可用")
            return tokenizer, True
        except:
            print("✗ word_ids()方法不可用")
            
    except Exception as e:
        print(f"✗ 快速tokenizer加载失败: {e}")
    
    # 测试慢速tokenizer
    try:
        print("\n尝试加载慢速tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_path, use_fast=False)
        print("✓ 慢速tokenizer加载成功")
        return tokenizer, False
    except Exception as e:
        print(f"✗ 慢速tokenizer也加载失败: {e}")
        return None, False

if __name__ == "__main__":
    # 你的模型路径
    model_path = "G:\\自然语言处理\\自然语言处理2025-10-22\\bert-base-chinese\\bert-base-chinese"
    
    # 检查目录
    if check_model_directory(model_path):
        tokenizer, is_fast = test_tokenizer(model_path)
        if tokenizer:
            print(f"\n✓ tokenizer加载成功，类型: {'快速' if is_fast else '慢速'}")
        else:
            print("\n✗ 无法加载tokenizer")
    else:
        print("\n✗ 模型目录不完整")