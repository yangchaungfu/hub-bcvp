
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
sentence = "常经有意见分歧"

def all_cut(sentence, Dict):
    def backtrack(start, path, result):
        # 如果已经处理到句子末尾，将当前路径加入结果
        if start == len(sentence):
            result.append(path[:])  # 使用切片创建副本
            return
        
        # 尝试所有可能的切分长度
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            
            # 如果当前子串在词典中，继续递归
            if word in Dict:
                path.append(word)
                backtrack(end, path, result)
                path.pop()  # 回溯，移除最后添加的词
    
    result = []
    backtrack(0, [], result)
    return result

# 测试函数
target = all_cut(sentence, Dict)

# 打印所有切分结果
for i, segmentation in enumerate(target, 1):
    print(f"{i:2d}. {segmentation}")

print(f"\n总共找到了 {len(target)} 种切分方式")

