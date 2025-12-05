#week3作业

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
ans = []

def dfs(idx, tmp, sentence, Dict):
    if idx == len(sentence):
        ans.append(tmp[:])
        return
    for length in range(1, len(sentence) - idx + 1):
        if sentence[idx : idx + length] in Dict:
            tmp.append(sentence[idx : idx + length])
            dfs(idx + length, tmp, sentence, Dict)
            tmp.pop()
    

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    dfs(0, [], sentence, Dict)    
    return ans 


def isSame(a, b):
    for i in a:
        if i not in b:
            return False
    for i in b:
        if i not in a:
            return False
    return len(a) == len(b)

def main():
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
    res = all_cut(sentence, Dict)
    for i in res:
        print(i)
    print("res == target:", isSame(res, target))


if __name__ == '__main__':
    main()
