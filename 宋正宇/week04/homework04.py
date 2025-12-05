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
    n = len(sentence)
    target = []

    def cutOneByOne(start, temp):
        if n == start:
            target.append(temp.copy())
            return

        for end in range(start+1, n+1):
            if sentence[start:end] in Dict:
                temp.append(sentence[start:end])
                cutOneByOne(end,temp)
                temp.pop()
    cutOneByOne(0, [])
    return target


def all_cut_second(sentence, Dict):
    n = len(sentence)
    target = [[] for _ in range(n + 1)]
    target[0] = [[]]

    for i in range(1, n + 1):
        for j in range(i):
            if target[j]:
                substr = sentence[j:i]
                if substr in Dict:
                    for s_split in target[j]:
                        target[i].append(s_split + [substr])
    return target[n]


#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]

print(all_cut(sentence, Dict))
print(all_cut_second(sentence, Dict))
