#week3作业
from pygments.lexer import words

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
    target = []
    length = len(sentence)

    def back_track(start, subReult):

        if start == length:
            # 若已遍历完整个字符串，将当前路径加入结果
            target.append(subReult.copy())
            return
        # 尝试从start开始的所有可能前缀
        for end in range(start + 1, length + 1):
            substring = sentence[start:end]
            if substring in Dict:
                # 若前缀在词典中，继续递归处理剩余部分
                subReult.append(substring)
                back_track(end, subReult)
                subReult.pop()  # 回溯，移除最后添加的词语

    # 从索引0开始回溯
    back_track(0, [])
    return target

all_result = all_cut(sentence, Dict)
for item in all_result:
    print(item)

"""
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
"""
