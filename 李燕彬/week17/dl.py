

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
        if memory["available_nodes"]:
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
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    def calucate_sentence_score(self, query, sentence):
        #两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)


    def slot_filling(self, memory):
        #槽位填充
        hit_node = memory["hit_node"]
        if hit_node:
            node_info = self.all_node_info[hit_node]
            query = memory['query']
            for slot in node_info.get('slot', []):
                if slot not in memory:
                    slot_values = self.slot_info[slot]["values"]
                    if re.search(slot_values, query):
                        memory[slot] = re.search(slot_values, query).group()
        return memory

    def dst(self, memory):
        hit_node = memory["hit_node"]
        if hit_node:
            node_info = self.all_node_info[hit_node]
            slot = node_info.get('slot', [])
            for s in slot:
                if s not in memory:
                    memory["require_slot"] = s
                    return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        if memory["require_slot"] is None:
            #没有需要填充的槽位
            memory["policy"] = "reply"
            # self.take_action(memory)
            hit_node = memory["hit_node"]
            if hit_node:
                node_info = self.all_node_info[hit_node]
                # 保存当前节点作为商品推荐节点
                if "action" in node_info and "select 衣服" in "".join(node_info.get("action", [])):
                    memory["product_node"] = hit_node
                memory["available_nodes"] = node_info.get("childnode", [])
            else:
                memory["available_nodes"] = []
        else:
            #有欠缺的槽位
            memory["policy"] = "request"
            if memory["hit_node"]:
                memory["available_nodes"] = [memory["hit_node"]] #停留在当前节点
            else:
                memory["available_nodes"] = []
        return memory

    def nlg(self, memory):
        # 检查是否需要重新介绍商品
        if memory.get('query', '').strip() == "重新介绍下商品":
            if 'last_response' in memory:
                memory["response"] = memory['last_response']
                # 重新介绍商品后，继续当前的问答流程
                # 检查当前是否有需要填充的槽位
                if memory.get("require_slot"):
                    # 如果有需要填充的槽位，添加提示信息
                    slot = memory["require_slot"]
                    if slot and slot in self.slot_info:
                        # 简化提示，只显示问题，不显示可选项
                        query = self.slot_info[slot]["query"]
                        memory["response"] += "\n" + query
                return memory
            elif 'product_node' in memory:
                # 如果没有保存的响应，根据product_node重新生成
                node = memory["product_node"]
                if node in self.all_node_info:
                    node_info = self.all_node_info[node]
                    memory["response"] = self.fill_in_slot(node_info["response"], memory)
                    memory['last_response'] = memory["response"]
                    # 重新介绍商品后，继续当前的问答流程
                    # 检查当前是否有需要填充的槽位
                    if memory.get("require_slot"):
                        # 如果有需要填充的槽位，添加提示信息
                        slot = memory["require_slot"]
                        if slot and slot in self.slot_info:
                            # 简化提示，只显示问题，不显示可选项
                            query = self.slot_info[slot]["query"]
                            memory["response"] += "\n" + query
                    return memory
        
        #根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            if hit_node:
                node_info = self.all_node_info[hit_node]
                memory["response"] = self.fill_in_slot(node_info["response"], memory)
                # 只保存商品推荐的响应
                if "action" in node_info and "select 衣服" in "".join(node_info.get("action", [])):
                    memory['last_response'] = memory["response"]
                # 购买商品完成后，打印购买的商品信息
                if "action" in node_info and ("MAKE_PAYMENT" in node_info["action"] or "TAKE_ORDER" in node_info["action"]):
                    if 'last_response' in memory:
                        memory["response"] += "\n您购买的商品信息：" + memory['last_response']
            else:
                memory["response"] = "对话已结束，感谢您的使用。"
                # 标记对话结束
                memory["dialogue_ended"] = True
        else:
            #policy == "request"
            slot = memory["require_slot"]
            if slot and slot in self.slot_info:
                # 构建包含可选项的响应
                query = self.slot_info[slot]["query"]
                # 尝试从values中提取可选项
                values = self.slot_info[slot].get("values", "")
                if values:
                    # 简单解析values，假设它是一个正则表达式或逗号分隔的选项
                    # 这里简化处理，实际可能需要更复杂的解析
                    options = []
                    # 尝试从正则表达式中提取选项
                    if "|" in values:
                        # 假设values是类似 "选项1|选项2|选项3" 的格式
                        options = [opt.strip() for opt in values.split("|")]
                    elif "," in values:
                        # 假设values是类似 "选项1,选项2,选项3" 的格式
                        options = [opt.strip() for opt in values.split(",")]
                    
                    if options:
                        # 移除可能的正则表达式特殊字符
                        clean_options = []
                        for opt in options:
                            # 移除正则表达式的开始和结束标记
                            opt = opt.strip("^$")
                            # 移除可能的量词
                            opt = re.sub(r"[+*?]{1,2}", "", opt)
                            # 移除可能的字符类标记
                            opt = opt.strip("[]")
                            if opt:
                                clean_options.append(opt)
                        
                        if clean_options:
                            query += " 可选：" + "、".join(clean_options[:5])  # 最多显示5个选项
                memory["response"] = query
            else:
                memory["response"] = "抱歉，我没有理解您的意思。"
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        if node and node in self.all_node_info:
            node_info = self.all_node_info[node]
            for slot in node_info.get("slot", []):
                if slot in memory:
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
        # 检查是否退出
        if query.strip() in ["退出", "quit", "Quit", "QUIT"]:
            print("再见！")
            break
        # query = "你好"    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
        # 检查对话是否结束
        if memory.get("dialogue_ended"):
            print("对话已结束，脚本退出。")
            break
        





