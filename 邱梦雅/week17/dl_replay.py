

'''
任务型多轮对话系统
读取场景脚本完成多轮对话

编程建议：
1.先搭建出整体框架
2.先写测试用例，确定使用方法
3.每个方法不要超过20行
4.变量命名和方法命名尽量标准
'''

import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
    
    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")


    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in node['childnode']]
            

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        #三列：slot, query, values
        self.slot_info = {}
        #逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values
      
    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        #意图识别，匹配当前可以访问的节点
        query = memory['query']
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory


    def calucate_node_score(self, query, node):
        #节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calculate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    def calculate_sentence_score(self, query, sentence):
        #两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)


    # def calculate_sentence_score2(self, s1, s2):
    #     # 用编辑距离计算文本相似度
    #     m, n = len(s1), len(s2)
    #     max_len = max(m, n)
    #     if max_len == 0:
    #         return 1.0  # 两个空字符串视为完全相似
    #     dp = [[0] * (n + 1) for _ in range(m + 1)]
    #     for i in range(m + 1):
    #         dp[i][0] = i
    #     for j in range(n + 1):
    #         dp[0][j] = j
    #     for i in range(1, m + 1):
    #         for j in range(1, n + 1):
    #             if s1[i - 1] == s2[j - 1]:
    #                 d = 0
    #             else:
    #                 d = 1
    #             dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + d)
    #     return 1 - dp[m][n] / max_len

    def slot_filling(self, memory):
        #槽位填充
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, query):
                    # re.search() 或 re.match() 找到内容后，并不会直接给你字符串，而是给你一个“对象”（告诉你找到了、在哪找到的），而 .group() 就是用来从这个对象里把具体的文字内容“掏”出来的方法
                    memory[slot] = re.search(slot_values, query).group()   # 在字符串 query 中搜索符合正则表达式模式 slot_values 的第一个子串，并返回该子串的内容
        return memory

    def dst(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        # if node_info["id"] != "repeat":
        #     memory["replayed_query"] = None
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        if node_info["id"] == "replay":
            # 进入重播逻辑
            memory["policy"] = "replay"
            memory["available_nodes"] = memory["last_available_childnode"]
            # if memory["replayed_query"] is None:
            #     memory["replayed_query"] = memory["last_response"]   # 记录待重播的query
        elif memory["require_slot"] is None:
            #没有需要填充的槽位
            memory["policy"] = "reply"
            # self.take_action(memory)
            # hit_node = memory["hit_node"]
            # node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            #有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]] #停留在当前节点
        return memory

    def nlg(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        if memory["policy"] == "replay":
            # hit_node = memory["hit_node"]
            # node_info = self.all_node_info[hit_node]
            memory["response"] = node_info["response"] + memory["last_response"]
        #根据policy执行反问或回答
        elif memory["policy"] == "reply":
            # hit_node = memory["hit_node"]
            # node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
            memory["last_response"] = memory["response"]  # 记录上一轮非重播的回复
            memory["last_available_childnode"] = node_info.get("childnode", [])   # 记录最近一轮非重播节点的所有子节点
        else:
            #policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
            memory["last_response"] = memory["response"]   # 记录上一轮非重播的回复
            memory["last_available_childnode"] = node_info.get("childnode", [])   # 记录最近一轮非重播节点的所有子节点
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            template = template.replace(slot, memory[slot])
        return template

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory
        

if __name__ == '__main__':
    ds = DialogueSystem()
    # print(ds.all_node_info)
    print(ds.slot_info)

    memory = {"available_nodes":["scenario-买衣服_node1","scenario-看电影_node1"]} #用户状态
    while True:
        query = input("请输入：")
        # 检测退出命令
        if query.lower() in ['exit', 'quit', 'q']:
            print("程序已退出。")
            break
        # query = "你好"    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")

