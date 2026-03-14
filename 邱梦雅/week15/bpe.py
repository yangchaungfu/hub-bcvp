from collections import defaultdict

def get_pairs_count(tokens):
    stats = defaultdict(int)
    for i in range(len(tokens) - 1):
        stats[(tokens[i], tokens[i + 1])] += 1
    return stats

def merge_pairs(tokens, pair, new_token_id):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i <  len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_token_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def encode(text, vocab_table):
    tokens = list(text.encode("utf-8"))  # 0~255原始数字序列
    # 根据词表对原始数字序列做替换（压缩）
    while len(tokens) >= 2:
        stats = get_pairs_count(tokens)
        # 在 stats 的所有元素中，找出那个在词表里对应token_id的值最小的二元组
        pair = None
        min_idx = float("inf")
        for p, count in stats.items():
            idx = vocab_table.get(p, float("inf"))
            if idx < min_idx:
                pair = p
                min_idx = idx
        # 如果stats里不存在词表vocab_table中的二元组，直接停止替换
        if pair is None:
            break
        # idx = vocab_table[pair]
        tokens = merge_pairs(tokens, pair, min_idx)
    return tokens

def decode(encoded_tokens, vocab_table):
    # vocab = {idx: bytes([idx]) for idx in range(256)}
    # 先在0~255的基础上添加词表中的新token_id
    vocab = defaultdict(bytes)
    for idx in range(256):
        vocab[idx] = bytes([idx])
    for (p0, p1), idx in vocab_table.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    tokens = b"".join(vocab[idx] for idx in encoded_tokens)   # b"".join(...)：将所有bytes对象拼接成一个完整的bytes
    text = tokens.decode("utf-8", errors="replace")    # errors="replace", 替换字符。将无法解码的字节替换为 ``
    return text


if __name__ == "__main__":

    text = (
            "《金偶像谜案》，原名《The Case of the Golden Idol》，是由Color Gray Games开发的一款推理游戏。该工作室已经推出其续作《金偶像崛起》，最近推出dlc，两款游戏都是5折新史低。"
            "《金偶像谜案》本体讲述了12个发生在18世纪，前后跨越50年的死亡案件，这些案件都与从异国而来的金偶像有着密不可分的关系，案件逐步深化，最后上升至政治层面。"
            "《金偶像谜案》的两个dlc《兰卡的蜘蛛》《里莫利亚吸血鬼》补全了本体的剧情，相当于本体的前传，使得本作的剧情完成闭环。"
            
            "《金偶像谜案》的推理过程是获取信息的过程，玩家在固定的场景通过与物体，人物互动获取信息，大部分信息会以关键词的形式储存，用以填写卷轴。《金偶像谜案》的目标就是填写卷轴，卷轴大体可以概括为几个部分："
            "案件综述，人物身份，案情相关内容。填写的内容不分先后顺序，这意味着玩家可以按照个人思考方式进行游戏。当一个部分的所有空白都被填写后，游戏会提示当前部分是否存在错误，分别为：卷轴填写存在错误和卷轴填写存在至多两个错误，"
            "玩家可以根据提示完善推理内容，同时，游戏还有提示系统，可以进一步简化内容。"
            
            "《金偶像谜案》的游戏方式简化了正常推理游戏中玩家获取信息的过程，仅需完全浏览完场景信息就可以得到全部关键词。《金偶像谜案》的推理过程与《奥伯拉丁的回归》有相似之处，"
            "都是在固定的场景之中搜寻信息，并且填写的内容本身带有一定的提示。《金偶像谜案》的信息获取除开可以“收到背包”中的关键词外，场景、人物、物件本身也带有线索，这使得《金偶像谜案》逻辑更加完备。"
            "此外，在游戏过程中，当你填写80%的内容后，如果没有进一步的思路，可以采用枚举法试出答案，之后可以再去理清思路。在游玩过程中，玩家面临的最大问题可能是人名的记忆，大量人名非常容易记不清，"
            "可能会需要反复确认。不过《金偶像谜案》的游玩时长并不长，本体加全dlc的游玩时长大概在6到10小时，在玩家对人名与大场景调查感到厌烦之前，游戏已经将内容讲述完毕。")

    # 1. utf8编码
    tokens = text.encode("utf-8")  # raw bytes
    tokens = list(tokens)
    print("原文长度：", len(text))
    print(tokens)
    print("utf8编码后长度：", len(tokens))
    print("=" * 60)

    # 2. 统计每个2元组出现次数
    stats = get_pairs_count(tokens)
    print("统计每个2元组出现次数：", sorted(stats.items(), key=lambda x: x[1], reverse=True))
    print("=" * 60)

    # 3. 选取出现次数最多的2元组，合并，构建词表（替换映射表）
    vocab_size = 300  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256  # 要合并的次数
    compressed_tokens = tokens.copy()   # 复制一份

    merge_table = defaultdict(int)  # bpe合并（压缩）过程中形成的词表（替换映射表）
    for i in range(num_merges):
        stats = get_pairs_count(compressed_tokens)
        # 找出当前统计次数最高的二元组
        max_freq_pair = max(stats, key=stats.get)
        # 替换
        new_idx = 256 + i
        print(f"把二元组 {max_freq_pair} 替换为 {new_idx}")
        compressed_tokens = merge_pairs(compressed_tokens, max_freq_pair, new_idx)
        merge_table[max_freq_pair] = new_idx   # 记录词表

    print("---")
    print("原始tokens长度:", len(tokens))
    print("bpe压缩后tokens长度:", len(compressed_tokens))
    print(f"压缩比率: {len(tokens) / len(compressed_tokens):.2f}X")
    print("==" * 60)

    # 4. 得到词表 merge_table，根据词表尝试将一句文本encode编码为0~255数字的表示序列
    sentence = "《金偶像谜案》的推理过程与《奥伯拉丁的回归》有相似之处，都是在固定的场景之中搜寻信息，并且填写的内容本身带有一定的提示。《金偶像谜案》的信息获取除开可以“收到背包”中的关键词外，场景、人物、物件本身也带有线索，这使得《金偶像谜案》逻辑更加完备。"
    # sentence = "Hello World!"
    # sentence = "𠮷𬺓覅梻𪜀𫜴𫠠_(:з」∠)_🏨𠑊嘦烎麤垚靐"
    encoded_tokens = encode(sentence, merge_table)
    print("原文:", sentence)
    print("encoded_tokens:", encoded_tokens)
    print("=" * 60)

    # 5. 根据词表 merge_table，再将压缩后的tokens 0~255数字序列还原为原文
    decoded_sentence = decode(encoded_tokens, merge_table)
    print("encoded_tokens:", encoded_tokens)
    print("还原:", decoded_sentence)
    print("=" * 60)

    sentence2 = decode(encode(sentence, merge_table), merge_table)
    print(sentence2 == sentence)
