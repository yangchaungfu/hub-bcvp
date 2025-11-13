#week4作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
       # 将字典转换为集合，提高查找效率
    word_set = set(Dict.keys())

       # 使用动态规划 + 回溯的方法
       # memo[i] 存储从位置 i 到末尾的所有切分方式
    memo = {}

    def dfs(start):
    # 从start位置开始，返回所有可能的切分方式
        if start in memo:
            return memo[start]

    # 如果已经到达句子末尾，返回空列表的列表（表示一种切分方式）
        if start == len(sentence):
            return [[]]

        result = []

    # 尝试从 start 位置开始的所有可能的词
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in word_set:
            # 如果当前词在字典中，递归处理剩余部分
                remaining_cuts = dfs(end)
            # 将当前词与剩余部分的每种切分方式组合
                for remaining in remaining_cuts:
                    result.append([word] + remaining)

        memo[start] = result
        return result

    return dfs(0)

# 测试
target = all_cut(sentence, Dict)

# 输出结果，验证是否正确
for idx, cut in enumerate(target):
    print(f"{idx+1}: {cut}")

# 验证结果数量和内容
expected_count = 14
print(f"\n总共找到 {len(target)} 种切分方式")
print(f"预期数量: {expected_count}")
print(f"结果正确: {len(target) == expected_count}")

# 检查是否包含所有预期结果
expected_results = [
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

all_found = True
for expected in expected_results:
    if expected not in target:
        print(f"缺失结果: {expected}")
        all_found = False

if all_found:
    print("所有预期结果都找到！")
else:
    print("部分预期结果未找到！")




