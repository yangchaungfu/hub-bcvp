"""

week3作业
实现句子全切分

"""

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
DICT = {"经常": 0.1,
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
SENTENCE = "经常有意见分歧"

# 目标输出;顺序不重要
TARGET = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]


# 深度优先
def dfs_find_word(sentence: str, word_set: set, start: int, path: list, results: list):
    if start == len(sentence):
        results.append(path.copy())
        return
    # 正向最大匹配
    for end in range(start + 1, len(sentence) + 1):
        word = sentence[start:end]
        if word in word_set:
            path.append(word)
            dfs_find_word(sentence, word_set, end, path, results)
            path.pop()  # 回溯


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence: str, word_dict: dict) -> list[list[str]]:
    word_set = set(word_dict)
    results = []
    dfs_find_word(sentence, word_set, 0, [], results)
    return results

if __name__ == '__main__':
    target = all_cut(SENTENCE, DICT)
    for tar in target:
        print(tar)
