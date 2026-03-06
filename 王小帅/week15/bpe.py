
import os


def get_stats(tokens):
    """统计相邻字符对的频率"""
    counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        # 先判断键是否存在，不存在则初始化为0
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """合并字节对"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def encode(text, vocab_size=300):
    """
    编码文本并训练BPE
    :param text: 原始文本
    :param vocab_size: 目标词汇表大小（≥256）
    :return: 编码后的ID列表、词汇表（合并规则）、完整映射表
    """
    # 初始化为UTF-8字节的整数列表
    tokens = text.encode("utf-8")
    ids = list(map(int, tokens))
    merges = {}  # 记录合并规则：(p0, p1) -> new_idx
    vocab = {idx: bytes([idx]) for idx in range(256)}  # 完整映射表：idx -> bytes

    # 计算需要合并的次数（目标词汇表大小-基础256个字节）
    num_merges = vocab_size - 256
    if num_merges <= 0:
        print("词汇表大小需≥256，直接返回原始ID")
        return ids, merges, vocab

    # 核心BPE合并逻辑
    for i in range(num_merges):
        # 统计当前字节对频率
        stats = get_stats(ids)
        if not stats:  # 无更多可合并的字节对，提前终止
            print(f"\n提前停止：第{i+1}次合并，无更多可合并的字节对")
            break
        # 找出频率最高的字节对
        pair = max(stats, key=stats.get)
        new_idx = 256 + i
        # 记录合并规则和映射表
        merges[pair] = new_idx
        vocab[new_idx] = vocab[pair[0]] + vocab[pair[1]]
        # 合并字节对
        ids = merge(ids, pair, new_idx)
        # 可选：打印合并过程（便于调试）
        # print(f"第{i+1}次合并：{pair} -> {new_idx} (映射为: {vocab[new_idx]})")

    return ids, merges, vocab


def decode(encoded_tokens, merges):
    """
    解码BPE编码结果
    :param encoded_tokens: 编码后的ID列表
    :param merges: 合并规则字典 {(p0,p1): new_idx}
    :return: 还原后的文本
    """
    # 第一步：反向构建映射表（new_idx -> (p0,p1)）
    reverse_merges = {new_idx: pair for pair, new_idx in merges.items()}
    # 第二步：将编码ID还原为原始字节ID列表
    ids = list(encoded_tokens)
    i = 0
    while i < len(ids):
        idx = ids[i]
        if idx in reverse_merges:
            # 替换合并后的ID为原始字节对
            ids.pop(i)
            ids.insert(i, reverse_merges[idx][1])
            ids.insert(i, reverse_merges[idx][0])
            # 替换后回退一位，检查新的相邻对
            i = max(0, i - 1)
        else:
            i += 1
    # 第三步：将原始字节ID转换为bytes并解码为文本
    try:
        tokens = bytes(ids)
        text = tokens.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"解码出错：{e}")
        text = ""
    return text


def encode_main(input_file="corpus.txt", vocab_size=300):
    """
    BPE编码/训练主入口
    :param input_file: 输入文本文件
    :param vocab_size: 目标词汇表大小
    :return: 编码ID列表、合并规则、完整映射表
    """
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 '{input_file}' 不存在！")
        return None, None, None

    print("=" * 60)
    print(f"BPE编码/训练系统 - 处理文件: {input_file}")
    print(f"目标词汇表大小: {vocab_size}")
    print("=" * 60)

    # 1. 读取原始文本
    with open(input_file, 'r', encoding='utf-8') as f:
        original_text = f.read().strip()  # 去除首尾空白（可选）

    print(f"原始文本长度: {len(original_text)} 字符")
    original_bytes = original_text.encode("utf-8")
    print(f"原始字节数: {len(original_bytes)}")

    # 2. 执行BPE编码和训练
    print("\n开始BPE训练/编码...")
    encoded_ids, merges, vocab = encode(original_text, vocab_size)

    print(f"\n编码完成！")
    print(f"编码后的ID数量: {len(encoded_ids)} (压缩率: {len(encoded_ids)/len(original_bytes):.2f})")

    # 3. 显示部分编码结果
    print("\n=== 部分编码结果（前50个ID） ===")
    for i in range(0, min(50, len(encoded_ids)), 10):
        chunk = encoded_ids[i:i + 10]
        print(f"ID {i:3}-{i + len(chunk) - 1:3}: {chunk}")

    # 4. 保存编码结果
    encoded_file = "corpus_encoded.txt"
    with open(encoded_file, 'w', encoding='utf-8') as f:
        f.write(','.join(map(str, encoded_ids)))
    print(f"\n编码结果已保存到: {encoded_file}")

    # 5. 保存词汇表（合并规则）
    vocab_file = "bpe_vocabulary.txt"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write("BPE词汇表（合并规则）\n")
        f.write("=" * 40 + "\n")
        f.write(f"基础字节数: 256\n")
        f.write(f"合并次数: {len(merges)}\n")
        f.write(f"最终词汇表大小: {256 + len(merges)}\n\n")

        # 保存合并规则
        f.write("合并规则 (按学习顺序):\n")
        for i, (pair, new_idx) in enumerate(merges.items()):
            byte_str = vocab[new_idx].decode('utf-8', errors='replace')
            f.write(f"{i+1:3}. ({pair[0]:3}, {pair[1]:3}) -> ID {new_idx:3} (对应: '{byte_str}')\n")

    print(f"词汇表已保存到: {vocab_file}")
    print("=" * 60)
    return encoded_ids, merges, vocab


if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "corpus.txt"  # 输入文本文件（需提前创建）
    VOCAB_SIZE = 300  # 目标词汇表大小（≥256）

    # 1. 执行编码/训练
    encoded_ids, merges, vocab = encode_main(INPUT_FILE, VOCAB_SIZE)

    # 2. 测试解码（仅当编码成功时）
    if encoded_ids and merges:
        print("\n=== 测试解码 ===")
        decoded_text = decode(encoded_ids, merges)
        print(f"解码结果（前200字符）: {decoded_text[:200]}...")
        # 验证：原始文本前200字符 vs 解码后前200字符
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            original_text = f.read().strip()
        print(f"原始文本（前200字符）: {original_text[:200]}...")
