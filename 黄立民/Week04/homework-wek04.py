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
    #TODO
    lens = len(sentence);
    # print(lens)
    words_all = [[] for i in range(lens+1)]
    words_all[-1] = [[]]

    max_len = max(len(word) for word in Dict)

    # for j in range(6 + 1, 7):
    #     print(j)
    #
    # for  i in range(lens-1,-1,-1):
    #     max_j = min(lens,i+max_len)
    #     for j in range(i+1,max_j+1):
    #         # print(i,j,lens)
    #         word = sentence[i:j]
    #         # print(word)
    #         if word in Dict:
    #             for rest in words_all[j]:
    #                 words_all[i].append([word]+rest)


    for i in range(lens,-1, -1):
        max_j = min(max_len,lens-i)
        # print((max_len,lens-i))
        for j in range(i+1,i+max_j+1,1):
            word = sentence[i:j]
            # print(i,j,word)
            if word in Dict:
                for rest in words_all[j]:

                    words_all[i].append([word] + rest)

    # return words
    return words_all[0]

words_all = all_cut(sentence,Dict)
lens = len(words_all)
for i in range(lens):
    print(words_all[i])

# print(all_cut(sentence,Dict))

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

