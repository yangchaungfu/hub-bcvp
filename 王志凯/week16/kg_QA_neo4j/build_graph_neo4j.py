# -*- coding:utf-8 -*-

import re
import json
from py2neo import Graph
from collections import defaultdict


class Build_Graph:
    def __init__(self, config):
        self.config = config
        self.graph = Graph(config["neo4j_url"], auth=(config["neo4j_username"], config["neo4j_password"]))
        self.relation_data = defaultdict(dict)
        self.attribute_data = defaultdict(dict)
        self.label_data = defaultdict(dict)
        self.load()
        self.create_graph()

    def load(self):
        # 处理实体-关系-实体
        with open(self.config["ent_rel_ent_data_path"], 'r', encoding='utf-8') as f:
            for line in f:
                head, relation, tail = line.split()
                head = self.handle_label(head)
                tail = self.handle_label(tail)
                self.relation_data[head][relation] = tail
        # 处理实体-属性-属性值
        with open(self.config["ent_attr_val_data_path"], 'r', encoding='utf-8') as f:
            for line in f:
                entity, attribute, value = line.split()
                entity = self.handle_label(entity)
                self.attribute_data[entity][attribute] = value

    # 提取所有的标签，并处理实体括号
    def handle_label(self, entity):
        # 匹配括号里面的标签（歌曲、专辑、电影、演唱会）
        match_res = re.search("（.+）", entity)
        if match_res:
            # 去掉括号
            label = match_res.group()[1:-1]
            if label in ["歌曲", "专辑", "电影", "演唱会"]:
                # 将括号内容去掉
                entity = re.sub("（.+）", "", entity)
                # 有些歌曲与专辑同名，故所有专辑使用《》区别开
                if label == "专辑":
                    entity = f"《{entity}》"
                self.label_data[entity] = label
        return entity


    # 创建图谱
    def create_graph(self):
        # 处理实体中带引号的问题
        def _safety(name: str):
            # 确保name中不包含反引号
            if '`' in name:
                raise ValueError(f"实体中不能包含反引号: {name}")
            return f"`{name}`"
        # 先清空数据
        self.graph.run("MATCH (n) DETACH DELETE n")

        # 存储所有的cypher语句及对应的参数，以元组形式（cypher，prop_dict）
        cypher_and_props = []
        graphed_entities = set()
        # 1.处理实体-属性-属性值
        for entity in self.attribute_data:
            # 构建实体所有属性字典（每个实体都添加名称属性）
            props = {"NAME": entity}
            props.update(self.attribute_data[entity])
            if entity in self.label_data:
                label = self.label_data[entity]
                cypher = f"CREATE ({_safety(entity)}:{label} $props)"
            else:
                cypher = f"CREATE ({_safety(entity)} $props)"
            cypher_and_props.append((cypher, {"props": props}))
            graphed_entities.add(entity)

        # 2.处理实体-关系-实体
        for head in self.relation_data:
            # 有些头实体只有关系，没有属性，需要单独给一个name属性
            if head not in graphed_entities:
                cypher = f"CREATE ({_safety(head)} $props)"
                cypher_and_props.append((cypher, {"props": {"NAME": head}}))
                graphed_entities.add(head)

            for relation, tail in self.relation_data[head].items():
                # 同样，有些尾实体也只有关系没有属性
                if tail not in graphed_entities:
                    cypher = f"CREATE ({_safety(tail)} $props)"
                    cypher_and_props.append((cypher, {"props": {"NAME": tail}}))
                    graphed_entities.add(tail)
                # 创建关系
                cypher = (f"MATCH ({_safety(head)} {{NAME: $head_name}}), ({_safety(tail)} {{NAME: $tail_name}}) "
                          f"CREATE ({_safety(head)})-[:{relation}]->({_safety(tail)})")
                cypher_and_props.append((cypher, {"head_name": head, "tail_name": tail}))
        # 执行所有语句
        for cypher, props in cypher_and_props:
            self.graph.run(cypher, **props)


    # 统计有哪些实体、属性、关系、标签
    def calc_count(self):
        data = defaultdict(set)
        for head in self.relation_data:
            data["entities"].add(head)
            for relation, tail in self.relation_data[head].items():
                data["entities"].add(tail)
                data["relations"].add(relation)
        for entity in self.attribute_data:
            data["entities"].add(entity)
            for attr in self.attribute_data[entity].keys():
                data["attributes"].add(attr)
        data["labels"] = ["歌曲", "专辑", "电影", "演唱会"]
        # 将 defaultdict(set) 转为 dict 类型
        data = dict((x, list(y)) for x, y in data.items())
        # 写入到文件kg_schema.json
        with open(self.config["schema_save_path"], 'w', encoding='utf-8') as f:
            # ensure_ascii=False: 不转义非ASCII字符
            # indent=4：json结构中的每层缩进用4个空格
            f.write(json.dumps(data, ensure_ascii=False, indent=4))



if __name__ == '__main__':
    from config import Config

    bg = Build_Graph(Config)
    bg.calc_count()
