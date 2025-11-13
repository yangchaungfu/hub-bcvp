import math

# 词典；每个词后方存储的是其词频
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
    memo={}
    def cut(s):
        if s in memo:
            return memo[s]
        if not s:
            return [[]]
        all_path=[]
        for i in range(len(s)+1):
            first_word=s[:i]
            if first_word in Dict:
                remain=s[i:]
                paths=cut(remain)
                for path in paths:
                   all_path.append([first_word]+path)
        memo[s]=all_path
        return all_path
    return cut(sentence)


def find_best_path(all_paths, Dict):
    """
    从所有切分路径中，找出“总词频最高”的路径
    """
    best_path = []
    # 我们要找“最大”的概率，所以初始值设为负无穷
    # 我们使用 log 概率，所以初始值是 -inf
    best_score = -float('inf')
    for path in all_paths:
        current_score = 0
        for word in path:
            freq = Dict[word]
            current_score += math.log(freq)
        if current_score > best_score:
            best_score = current_score
            best_path = path
    return best_path, best_score
# 1. 实现全切分
target = all_cut(sentence, Dict)

for t in target:
    print(t)


best_path, best_score = find_best_path(target, Dict)

print(f"最佳路径: {' / '.join(best_path)}")
print(f"路径总分 (Log-Probability): {best_score:.4f}")
