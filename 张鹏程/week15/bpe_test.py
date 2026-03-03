def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
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


def main(corpus_path, vocab_size, vocab_path):
    merges = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
        print("text len:", len(text))

        tokens = list(map(int, text.encode('utf8')))
        print("utf8 tokens len:", len(tokens))

        # stats = get_stats(tokens)
        # print(max(stats, key=stats.get), "-", max(stats.values()))

        # 复制原列表
        ids = list(tokens)
        for i in range(256, vocab_size + 1):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            num = max(stats.values())
            if num == 1:
                break
            # print(f"merging {pair} - {num} into a new token {i}")
            ids = merge(ids, pair, i)
            merges[pair] = i

        print(f"ids len:{len(ids)}")
    if merges:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for pair, num in merges.items():
                f.write(f"{pair[0]}\t{num}\n")


if __name__ == '__main__':
    corpus_path = "./corpus.txt"
    vocab_size = 800
    vocab_path = "./vocab.txt"
    main(corpus_path, vocab_size, vocab_path)
