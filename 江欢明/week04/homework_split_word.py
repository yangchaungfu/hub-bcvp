# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    split_word(sentence, Dict, [])


#句子切词函数；传入句子，递归获取单词保存到storage_list中
def split_word(sentence, Dict, storage_list):
    if (sentence == ''):
        print(storage_list)
        return storage_list

    first_word = sentence[0:1]
    prefix_word_List = prefix_word_query(Dict, first_word)
    if (len(prefix_word_List) == 0):
        storage_list.append(first_word)
        split_word(sentence[1:len(sentence)], Dict, storage_list)
    else:
        for word in prefix_word_List:
            new_storage_list = list(storage_list)
            new_storage_list.append(word)
            split_word(sentence[len(word):len(sentence)], Dict, new_storage_list)
    return storage_list

# 根据前缀词查询字典中所有的存在的词列表
def prefix_word_query(Dict, first_word):
    prefix_word_List = [];
    for key in Dict:
        if (key.startswith(first_word)):
            prefix_word_List.append(key)
    return prefix_word_List;


if __name__ == "__main__":
    all_cut(sentence, Dict)
