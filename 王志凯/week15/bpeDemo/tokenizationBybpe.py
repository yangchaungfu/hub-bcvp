# -*- coding:utf-8 -*-


# 统计组合出现次数（pair频率），返回dict：k组合，v频率
def count_freq(utf8_code: list):
    freq_dict = {}
    for pair in zip(utf8_code, utf8_code[1:]):
        freq_dict[pair] = freq_dict.get(pair, 0) + 1
    return freq_dict

# 将原编码序列中频率最高的组合替换成新的token
def merge(utf8_code: list, pair: tuple, index: int):
    new_utf8_code = []
    i = 0
    while i < len(utf8_code):
        if i < len(utf8_code) - 1 and utf8_code[i] == pair[0] and utf8_code[i + 1] == pair[1]:
            new_utf8_code.append(index)
            i += 2
        else:
            new_utf8_code.append(utf8_code[i])
            i += 1
    return new_utf8_code

# 将新词表写入文件
def write_vocab(vocab: dict, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        for k, v in vocab.items():
            f.write(f"{k}\t{v}\n")
    return

def main(config: dict):
    data_path = config["data_path"]
    vocab_size = config["vocab_size"]
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
        org_utf8_code = list(text.encode('utf-8'))
        vocab = {}
        utf8_code = org_utf8_code
        for index in range(256, vocab_size + 1):
            freq_dict = count_freq(utf8_code)
            # 获取出现次数最多的组合
            pair, freq = max(freq_dict.items(), key=lambda x: x[1])
            # 如果所有组合出现频率都为1，则直接停止压缩
            if freq == 1:
                break
            # 使用新的token替换该组合，返回新的编码序列
            utf8_code = merge(utf8_code, pair, index)
            # 更新词表
            vocab[index] = pair
        print(f"原文档token数：{len(org_utf8_code)}")
        print(f"BPE之后token数：{len(utf8_code)}")
        print(f"压缩比：{len(org_utf8_code) / len(utf8_code):.2f}")
        # 将词表写入文档
        write_vocab(vocab, config["vocab_path"])


if __name__ == '__main__':
    config = {
        "data_path": "./data/corpus.txt",
        "vocab_path": "./data/vocab.txt",
        "vocab_size": 1000
    }
    main(config)