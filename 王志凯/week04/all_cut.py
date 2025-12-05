# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 12,
        "经": 1,
        "有": 3,
        "常": 2,
        "有意见": 345,
        "歧": 7,
        "意见": 45,
        "分歧": 67,
        "见": 5,
        "意": 4,
        "见分歧": 567,
        "分": 6}


# 获取词表中最大长度，并将词表中的keys取出来组成列表
def trans_dict(Dict):
    keys = list(Dict.keys())
    max_len_word = max(keys, key=len)
    return keys, len(max_len_word)


# 使用动态规划的思想进行切分
def all_cut(sentence, Dict):
    sent_len = len(sentence)
    # 将关键词字典转化为关键词列表，并返回最大长度
    word_dict, max_len = trans_dict(Dict)

    # 要针对不同长度的文本进行组合划分，最终结果其实是需要取target[7]，所以初始化长度为7
    cut_result = [[] for _ in range(sent_len + 1)]
    # 动态规划需要对空文本进行初始化组合，否则cut_result[0]=[]，这在后续没法遍历cut_result[0]
    cut_result[0] = [[]]

    # 要计算n个字的划分方式，其实依赖于前（n-1）个字的划分方式，所以需要对前面划分结果进行保留
    for i in range(1, sent_len + 1):
        # 最大划分单元不能超过最大词长度
        for j in range(1, max_len + 1):
            if i - j >= 0:
                # 从后往前依次进行截取，并判断是否包含在词典中
                word = sentence[i - j:i]
                if word not in word_dict:
                    continue
                # 取出前面不同的划分方式与新词进行重新组合
                for pre_word in cut_result[i - j]:
                    new_word = pre_word + [word]
                    # print(new_word)
                    cut_result[i].append(new_word)
    return cut_result


def main():
    sentence = "经常有意见分歧"
    cut_result = all_cut(sentence, Dict)
    # 最终结果取最后一个元素
    target = cut_result[len(sentence)]
    for i, v in enumerate(target):
        print(f"{i + 1}:{v}")


if __name__ == "__main__":
    main()

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


