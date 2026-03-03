def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train_bpe(file_path, vocab_size):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = list(text.encode("utf-8"))
    
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = {}
    
    num_merges = vocab_size - 256
    for i in range(num_merges):
        stats = get_stats(tokens)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"Merging {pair} into a new token {idx}")
        tokens = merge(tokens, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    
    return vocab, merges

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

# 训练 BPE 模型
vocab, merges = train_bpe("巫医.txt", vocab_size=300)

# 测试文本
test_text = "巫医是一个强大的辅助英雄，擅长控制和治疗。"

# 打印原始 UTF-8 编码
original_utf8_bytes = list(test_text.encode("utf-8"))
print("Original UTF-8 encoding:", original_utf8_bytes)

# 编码和解码
encoded = encode(test_text, merges)
print("Encoded:", encoded)

decoded = decode(encoded, vocab)
print("Decoded:", decoded)

# 验证解码结果
print("Original text equals decoded text:", test_text == decoded)

# 计算压缩率
original_length = len(original_utf8_bytes)
encoded_length = len(encoded)
compression_ratio = original_length / encoded_length
print(f"\nCompression Analysis:")
print(f"Original UTF-8 byte count: {original_length}")
print(f"BPE token count: {encoded_length}")
print(f"Compression ratio: {compression_ratio:.2f} (bytes per token)")
