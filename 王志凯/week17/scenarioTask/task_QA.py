# -*- coding:utf-8 -*-

"""
任务型问答系统：
    1. 输入：用户需求
    2. 意图识别、槽位填充（nlu）
    3. 状态追踪（dst）
    4. 策略优化（dpo）
    5. 任务执行（nlg）
"""

import re
import os
import json
import pandas as pd

class TaskQA:
    def __init__(self):
        self.load()

    def load(self):
        # 所有节点信息
        self.all_node_info = {}
        # 所有槽位信息
        self.slot_info = {}
        # 加载场景信息
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_scenario("scenario-订火车票.json")
        # 加载槽位信息
        self.load_slot_info("slot_fitting_templet.xlsx")
        # 加载重复话术场景语句
        self.load_repeat_scenario("repeat_scenario.json")

    def load_scenario(self, scenario_file):
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = os.path.basename(scenario_file).split(".")[0]
        for node in self.scenario:
            self.all_node_info[scenario_name + "-" + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + "-" + child for child in node["childnode"]]
    
    def load_slot_info(self, slot_file):
        slot_templet = pd.read_excel(slot_file)
        """
        总共三列：slot为key，query和values为value
        query: 槽位填充的查询语句
        values: 槽位填充的可能值
        """
        for _, row in slot_templet.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
                self.slot_info[slot]["query"] = query
                self.slot_info[slot]["values"] = values
    
    def load_repeat_scenario(self, scenario_file):
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.repeat_scenario = json.load(f)


    def handle_repeat_scenario(self, memory):
        memory["repeat"] = False
        query = memory["query"]
        # 检查是否是"请再说一遍"之类的请求
        if any(phrase in query for phrase in self.repeat_scenario):
            memory["repeat"] = True
            # 如果有上一步的回复，直接返回
            if "last_response" in memory:
                memory["response"] = memory["last_response"]
            else:
                # 如果没有上一步的回复，返回默认提示
                memory["response"] = "抱歉，我不太明白"
        return memory
        
    def nlu(self, memory):
        # 意图识别：计算用户输入与所有available_nodes的相似度，选择相似度最高的节点作为hit_node
        memory = self.calc_similarity(memory)
        # 槽位填充：从用户输入中提取槽位值
        memory = self.fill_slot(memory)
        return memory
    
    def calc_similarity(self, memory):
        query = memory["query"]
        available_nodes = memory["available_nodes"]
        # 调用方法，使用最简单的jaccard相似度计算用户输入与所有available_nodes的相似度
        max_score = -1
        hit_node = None
        for node in available_nodes:
            intents = self.all_node_info[node]["intent"]
            for intent in intents:
                score = self.jaccard_similarity(query, intent)
                if score > max_score:
                    max_score = score
                    hit_node = node
        memory["hit_node"] = hit_node
        return memory

    def jaccard_similarity(self, str1, str2):
        set1 = set(str1)
        set2 = set(str2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def fill_slot(self, memory):
        query = memory["query"]
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        # 存储已匹配的文本，避免重复匹配
        matched_text = set()
        
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                match = re.search(slot_values, query)
                if match:
                    # 提取匹配的文本
                    matched_value = match.group()
                    # 检查是否已经匹配过相同的文本
                    if matched_value not in matched_text:
                        memory[slot] = matched_value
                        matched_text.add(matched_value)
        return memory

    def dst(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slots = node_info.get('slot', [])
        # 找到第一个未填充的槽位
        for slot in slots:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        # 所有槽位都已填充
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        if memory["require_slot"] is None:
            # 没有需要填充的槽位
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            # 有欠缺的槽位，继续请求槽位
            memory["policy"] = "request"
            # 停留在当前节点
            memory["available_nodes"] = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        if memory["policy"] == "reply":
            """
            槽位填充完成：根据模板生成回复
            """
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            # 将回复模板中的槽位填充成具体的值
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            """
            有欠缺的槽位，继续请求槽位
            """
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
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
        memory: 全局状态
        '''
        memory["query"] = query
        # 判断是否需要重复上一步
        memory = self.handle_repeat_scenario(memory)
        if memory["repeat"]:
            return memory
        # 自然语言理解：意图识别和槽位填充
        memory = self.nlu(memory)
        # 状态追踪：根据用户输入和当前节点，更新require_slot
        memory = self.dst(memory)
        # 策略优化：根据require_slot更新policy和available_nodes
        memory = self.dpo(memory)
        # 自然语言生成：根据policy和槽位值，选择回复或请求槽位
        memory = self.nlg(memory)
        # 存储当前回复，以便后续可以重复
        memory["last_response"] = memory["response"]
        return memory


if __name__ == "__main__":
    task_qa = TaskQA()
    memory = {"available_nodes": ["scenario-买衣服-node1", "scenario-看电影-node1", "scenario-订火车票-node1"]}
    while True:
        query = input("用户：")
        memory = task_qa.run(query, memory)
        print("AI：" + memory["response"])