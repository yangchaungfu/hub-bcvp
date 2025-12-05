# week4作业
import json

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
# 递归方式实现
def all_cut(sentence, Dict):
    target = []

    def backtrace(start, path):
        # 若到达句子末尾，则结束处理
        if start == len(sentence):
            target.append(path.copy())  # 使用copy防止后续更新
            return
        # 从start开始所有的子字符串
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 若子串在词典中则继续递归
            if word in Dict:
                path.append(word)
                backtrace(end, path)
                path.pop()  # 回溯，移除最后添加的值

    backtrace(0, [])
    return target


# 动态规划方式实现
def all_cut1(sentence, Dict):
    n = len(sentence)
    # dp[i]存储sentence[:i]的所有切分
    dp = [[] for _ in range(n + 1)]
    # 空字符串的切分结果为空列表，初始化列表
    dp[0] = [[]]
    # 计算dp[1]到dp[n]（前1个字符到前n个字符的切分结果）
    for i in range(1, n + 1):
        # 遍历所有可能得起始位置
        for j in range(i):
            word = sentence[j:i]
            # 若在词典中，且dp[j]有有效切分，则组合结果
            dpj = dp[j]
            if word in Dict and dpj:
                # 将dp[j]中每个切分结果加上当前词，添加到dp[i]中
                for path in dpj:
                    new_path = path.copy()
                    new_path.append(word)
                    dp[i].append(new_path)
    return dp[n]


# print(all_cut(sentence, Dict))
print(all_cut1(sentence, Dict))
# 目标输出;顺序不重要
target = [
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
