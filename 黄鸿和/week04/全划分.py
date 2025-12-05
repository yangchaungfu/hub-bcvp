#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
#日期：2025-11-05
# 作者：黄鸿和
from math import e


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
    #TODO
    word_set = set(Dict.keys())
    n = len(sentence)
    memory = {}

    # 深度优先搜索
    def dfs(i):
        # 递归出口
        if i == n:
            return [[]]
        # 序号相同，划分结果也一致，直接返回即可
        if i in memory:
            return memory[i]
        
        result = []

        for j in range(i + 1, n + 1):
            words = sentence[i:j]
            if words in word_set:
                # 得到该分支开头为words的所有可能的切分方式
                end_results = dfs(j)
                # 讲words 开头 与 每个后缀结果进行拼接，作为一个结果
                for end_result in end_results:
                    result.append([words] + end_result)
        
        memory[i] = result
        return result
    
    result = dfs(0)
    return result





#目标输出;顺序不重要
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


print(all_cut(sentence, Dict))

print('len(all_cut(sentence, Dict))', len(all_cut(sentence, Dict)))
print('len(target)', len(target))


# 将内层列表转换为元组（因为列表不可哈希，不能放入set中）
result_set = set(tuple(item) for item in all_cut(sentence, Dict))
target_set = set(tuple(item) for item in target)

print('\n\n')
print('result_set', result_set)
print('\n\n')
print('target_set', target_set)

if result_set == target_set:
    print("Yes")
else:
    print("No")
