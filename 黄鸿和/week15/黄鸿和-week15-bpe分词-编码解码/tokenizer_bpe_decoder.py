

with open('merges.txt', 'r', encoding='utf-8') as f:
    merges = eval(f.read())
    
with open('ids.txt', 'r', encoding='utf-8') as f:
    ids = eval(f.read())
    
# print('merges', merges)
# print('ids', ids[:300])

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    # print('p0', p0, 'p1', p1, 'idx', idx)
    # b'Hello, ' + b'world'   # 结果是 b'Hello, world'
    # b'\x01\x02' + b'\x03'   # 结果是 b'\x01\x02\x03'
    vocab[idx] = vocab[p0] + vocab[p1]
    print('vocab[idx]', vocab[idx])

def decode(ids):
	# given ids (list of integers), return Python string
#   ids：一串 token id（整数列表），比如 [300, 258, 97, ...]。
# 	vocab[idx]：每个 id 对应的 字节串 bytes，前面通过 vocab = {} 和 BPE 合并已经构建好了。
# 	b"".join(...)：用空的字节串作为分隔符，把所有 vocab[idx] 串起来，得到一个完整的 UTF-8 字节流。
#   tokens == b'A\xe4\xbd\xa0B'
  tokens = b"".join(vocab[idx] for idx in ids)
# .decode("utf-8")：按 UTF-8 编码规则，把 bytes 转成 Python 的 字符串 str（人类可读文本）。errors="replace"：如果有非法/不完整的 UTF-8 字节，不报错，而是用 替换。
  text = tokens.decode("utf-8", errors="replace")
  return text

print(decode(ids))