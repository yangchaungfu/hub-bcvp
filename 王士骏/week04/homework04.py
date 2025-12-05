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

def prefix_word_dict(dict):
    prefix_dict = {}
    for key in dict:
        for i in range(1, len(key)):
            if key[:i] not in prefix_dict: #不能用前缀覆盖词
                prefix_dict[key[:i]] = 0  #前缀
        prefix_dict[key] = 1  #词
    return prefix_dict

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    target = []
    if not sentence:
        return [[]]
    if sentence[0] not in Dict:
        for i in all_cut(sentence[1:],Dict):
            target.append([sentence[0]] + i)
    else:
        j = 1
        while sentence[:j] in Dict and j <= len(sentence):
            if Dict[sentence[:j]] == 1:
                for i in all_cut(sentence[j:], Dict):
                    target.append([sentence[:j]] + i)
            j = j + 1
    return target

# 目标输出;顺序不重要
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

result = all_cut(sentence,prefix_word_dict(Dict))
print("'{0}'这句话共有 {1} 种分词结果".format(sentence,len(result)))
for i, j in enumerate(result):
    print(i+1,":",j)
