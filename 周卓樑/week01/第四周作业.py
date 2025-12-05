# 词典
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"


def all_cut(sentence, Dict):
    results = []

    def dfs(start, path):
        # 如果到达句子末尾，记录结果
        if start == len(sentence):
            results.append(path[:])
            return

        # 从当前位置开始，尝试所有可能的子串
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                dfs(end, path)
                path.pop()  # 回溯

    dfs(0, [])
    return results


# 测试
target = all_cut(sentence, Dict)
for t in target:
    print(t)
