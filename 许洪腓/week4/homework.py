#week4作业
import random
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
    # # 方法一：回溯算法
    
    # sentence_len = len(sentence)
    # path = []
    # result = []
    # def backtracking(start):
    #     if start>=sentence_len:
    #         result.append(path.copy())
    #         return 
    #     for end in range(start+1,sentence_len+1):
    #         word = sentence[start:end]
    #         if word in Dict:
    #             path.append(word)
    #             backtracking(end)
    #             path.pop()
    # backtracking(0)
    # print(len(result))
    # return result

    # 方法二：动态规划
    '''
    1.dp[i]，前n个字符串的所有分类
    2.dp[i]= dp[i-j]的各个分类 + [word]，j满足sentence[j:i] in Dict
    3.初始化：dp[0]=[[]]
    '''
    sentence_len = len(sentence)
    dp = [[] for _ in range(sentence_len+1)]
    dp[0] = [[]]
    for i in range(1,sentence_len+1):
        for j in range(1,i+1):
            if sentence[i-j:i] in Dict:
                for path in dp[i-j]:
                    dp[i].append(path+[sentence[i-j:i]])
    return dp[-1]

def main():
    target = all_cut(sentence,Dict)
    for each in target:
        print(each)

if __name__ == '__main__':
    main()

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

