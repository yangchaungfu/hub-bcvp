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
    result = []# 分词结果集
    iterSentenceProcess(0, [], result)
    return result
# 循环遍历词典与文本进行比对，有匹配的则放入wordArr中，最后统一返回
def iterSentenceProcess(startIndex, wordArr, result):
    n = len(sentence) #句子的长度  7
    max_word_legth = max(len(word) for word in Dict) # 词典中的最长单词长度  3
    if startIndex >= n:
        result.append(wordArr.copy())  # 当遍历索引大于或等于文本长度时，代表结束，将当前切分后的语序加入最终结果集，不用copy会导致最终结果都为最后一次拆分结果（空数组）
        # print(path.copy())
        return
    for i in range(startIndex + 1, min(startIndex + max_word_legth + 1, n + 1)):  # start=0, end=4 --> range(1,4) 
        word = sentence[startIndex:i]  # word = sentence[0,1]-->经  将文本进行拆分
        # print(word)
        if word in Dict:
            wordArr.append(word) # 将匹配到的元素添加到子集合中
            # print(path)
            iterSentenceProcess(i, wordArr, result) #递归调用，继续遍历
            wordArr.pop() #删除并返回最后一个元素，回到上一次的结果状态
            # print(path)
if __name__ == '__main__':
    targetRes = all_cut(sentence, Dict)
    for res in list(targetRes):
        print(f"{res}")
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


