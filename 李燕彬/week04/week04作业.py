# coding: utf8

"""
基于给定的词表，最大概率进行中文分词
"""

dictory = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.01,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1
}


def split_word(sentence) -> list:
    result = []
    total = len(sentence)
    last_word = ''
    for k in range(total):
        j = total  # 从后往前匹配, 每次从句尾最后一个数字开始
        match_words = []  # 从当前k位置开始匹配到的词，可能有多个
        while j > k:
            # print(k, j, sentence[k:j])
            w = sentence[k:j]
            if w in dictory:
                match_words.append(w)
            j-=1
        if k != 0:
            last_word = sentence[k-1]  #
        # print(k, match_words, result, last_word)
        if k == 0: # 第一个字，新建列表
            for w in match_words:
                result.append([w])
        else: # 非第一个字，基于上一个字的结果，扩展
            new_result = []
            for i in range(len(result)):
                if result[i][-1][-1] == last_word:
                    # 匹配到前一个词，扩展数组
                    for w in match_words:
                        new_result.append(result[i] + [w])
                else:
                    # 未匹配到前一个词，保留原有词组
                    new_result.append(result[i])
            result = new_result
    return result

def calc_max_score_for_word(words: list) -> list:
    max_score = 0
    best_word = []
    score = 0
    for idx in range(len(words)):
        for w in words[idx]:
            score += dictory.get(w, 0.00001)
        if score > max_score:
            max_score = score
            best_word = words[idx]
        print(f"切分结果{idx+1}:\t{'|'.join(words[idx])},\t词频总和:  {score}")
    return max_score, best_word

if __name__ == "__main__":
    sentences = "经常有意见分歧"
    data = (split_word(sentences))
    print(f"总共有{len(data)}种切分结果, 词频最高结果是:{calc_max_score_for_word(data)}")
