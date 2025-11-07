#week3作业

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

    # 存储所有切分结果及对应的词频乘积
    all_results = []

    # 递归函数：寻找从start索引开始的所有切分方式
    def backtrack(start, path, current_product):
        # 若已到达句子末尾，记录当前切分路径和乘积
        if start == len(sentence):
            all_results.append((path.copy(), current_product))
            return
        # 尝试从start开始的所有可能长度的词
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 若词在词典中，继续递归
            if word in Dict:
                path.append(word)
                # 累积词频乘积（概率）
                backtrack(end, path, current_product * Dict[word])
                path.pop()  # 回溯

    # 从索引0开始，初始路径为空，初始乘积为1
    backtrack(0, [], 1)

    # 按词频乘积从大到小排序，再提取切分路径
    all_results.sort(key=lambda x: -x[1])
    target = [item[0] for item in all_results]
    return target


# 测试函数
result = all_cut(sentence, Dict)
# 打印结果（与预期目标对比）
for cut in result:
    print(cut)



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


