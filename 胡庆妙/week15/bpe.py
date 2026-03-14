
"""
实现bpe算法，编写一个支持任意语种的编码、解码方法
"""

from collections import defaultdict


def build_pair2idx(corpus_filename, vocab_size):
    """
    读取语料文本，构建高频字符对与索引的映射，直至总词表到达指定大小
    params:
        corpus_filename: 语料文本文件名
        vocab_size: 目标词表大小
    return:
        merged_pair_2_idx: 合并的字符对与索引的映射：(int, int) -> int
    """
    merged_pair_2_idx = {}  # 字符对与索引的映射：(int, int) -> int

    # 读取语料文本，将其转换为utf8编码
    with open(corpus_filename, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.encode("utf-8")  # 将文本转换为字节序列，文本中的每个字符由1~4个字节表示
    ids = list(map(int, tokens))  # 每个字节可表示为1个16进制数，或1个0~255之间的整数

    new_idx = 256  # 前面的0~255是utf8的单字节编码值
    while new_idx < vocab_size:
        # 统计相邻字符组合的出现频率
        pair2Cnt = stat_pairs(tokens)

        # 找到出现频率最高的字符组合
        top_pair = max(pair2Cnt, key=pair2Cnt.get)

        # 将字符组合替换为一个新的索引
        print(f"merging {top_pair} into a new token {new_idx}")
        tokens = merge(tokens, top_pair, new_idx)

        # 更新字符组合与索引的映射
        merged_pair_2_idx[top_pair] = new_idx
        new_idx += 1
    return merged_pair_2_idx


def stat_pairs(ids):
    pair2Cnt = defaultdict(int)
    for pair in zip(ids[:-1], ids[1:]):
        pair2Cnt[(pair[0], pair[1])] += 1
    return pair2Cnt


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def encode(text, idx_2_merged_pair):
    """
    根据索引与已合并字符对的映射，对文本进行解码
    """
    tokens = text.encode("utf-8")  # 将文本转换为字节序列，文本中的每个字符由1~4个字节表示
    ids = list(map(int, tokens))  # 每个字节可表示为1个16进制数，或1个0~255之间的整数

    for i in range(len(idx_2_merged_pair)):
        idx = 256 + i
        pair = idx_2_merged_pair[idx]
        j = 0
        while j < len(ids):
            if j < len(ids) - 1 and ids[j] == pair[0] and ids[j + 1] == pair[1]:
                ids[j] = idx
                del ids[j + 1]
            j += 1
    return ids


def decode(ids, idx_2_merged_pair):
    """
    根据索引与已合并字符对的映射，对编码序列进行解码
    """
    new_ids = []
    for idx in ids:
        if idx in idx_2_merged_pair.keys():
            pair = idx_2_merged_pair[idx]
            parse_merged_pair(pair, idx_2_merged_pair, new_ids)
        else:
            new_ids.append(idx)
    return bytes(new_ids).decode("utf-8")


def parse_merged_pair(merged_pair, idx_2_merged_pair, target_ids):
    """
    解析合并后的字符对，将其转换为原始的、值在0~255之间的整数序列
    """
    if merged_pair[0] < 256:
        target_ids.append(merged_pair[0])
    else:
        # 如果字符对中包含大于255的索引，则递归解析
        parse_merged_pair(idx_2_merged_pair[merged_pair[0]], idx_2_merged_pair, target_ids)

    if merged_pair[1] < 256:
        target_ids.append(merged_pair[1])
    else:
        # 如果字符对中包含大于255的索引，则递归解析
        parse_merged_pair(idx_2_merged_pair[merged_pair[1]], idx_2_merged_pair, target_ids)


if __name__ == "__main__":
    vocab_size = 300

    # 字符组合与索引的映射
    merged_pair_2_idx = build_pair2idx("bpe_corpus.txt", vocab_size)
    print("merged_pair_2_idx: ", merged_pair_2_idx)

    # 索引与字符组合的映射
    idx_2_merged_pair = {v: k for k, v in merged_pair_2_idx.items()}
    print("\nidx_2_merged_pair: ", idx_2_merged_pair)

    input_text = "你好，我是孙悟空。你是哪来妖精，看你往哪跑？"
    print("\nutf8 encode(input_text): ", list(input_text.encode("utf-8")))

    ids = encode(input_text, idx_2_merged_pair)
    print("\n bpe encode(input_text): ", ids)

    text = decode(ids, idx_2_merged_pair)
    print("\ntext: ", text)
