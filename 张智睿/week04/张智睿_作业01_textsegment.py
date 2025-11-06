from typing import List, Set

def all_cut(sentence: str, Dict: Set[str]) -> List[List[str]]:
    """
    实现文本全切分，返回所有可能的切分方式
    Input:
        sentence: 待切分文本
        Dict: 词典集合
    Returns:
        所有可能的切分方式列表
    """
    if not sentence:
        return [[]]

    # 计算最大词长
    max_word_len = max(len(word) for word in Dict) if Dict else 0
    results = []

    def dfs(start: int, path: List[str]):
        """
        DFS递归函数
        Args:
            start: 当前搜索起始位置
            path: 当前切分路径
        """
        # 终止条件：到达句子末尾
        if start == len(sentence):
            results.append(path[:])
            return

        # 尝试所有可能的词长
        for end in range(start + 1, min(start + max_word_len + 1, len(sentence) + 1)):
            word = sentence[start:end]

            # 如果词在词典中，继续搜索
            if word in Dict:
                path.append(word)
                dfs(end, path)
                path.pop()  # 回溯

    dfs(0, [])
    return results

def main():
    """主函数"""
    # 定义词典
    Dict = {"经常", "经", "有", "常", "有意见", "歧", "意见", "分歧",
            "见", "意", "见分歧", "分"}

    # 目标句子
    sentence = "经常有意见分歧"

    print("=" * 40)

    # 进行全切分
    results = all_cut(sentence, Dict)

    print(f"句子: '{sentence}'")
    print(f"共有 {len(results)} 种切分方式:")

    for i, seg in enumerate(results, 1):
        print(f"{i:2d}. {seg}")

if __name__ == "__main__":
    main()
