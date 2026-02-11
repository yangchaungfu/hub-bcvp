# -*- coding:utf-8 -*-

import itertools
import json
import re
import pandas
from py2neo import Graph


class GraphQA:
    def __init__(self, config):
        self.graph = Graph(config["neo4j_url"], auth=(config["neo4j_username"], config["neo4j_password"]))
        self.schema_path = config["schema_save_path"]
        self.templet_path = config["templet_path"]
        self.config = config
        self.load()

    def load(self):
        # 加载所有实体、关系、属性、标签
        self.load_kg_schema(self.schema_path)
        # 加载cypher模板
        self.load_templet(self.templet_path)

    def load_kg_schema(self, schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.entities = data["entities"]
            self.relations = data["relations"]
            self.attributes = data["attributes"]
            self.labels = data["labels"]

    def load_templet(self, templet_path):
        data_frame = pandas.read_excel(templet_path)
        self.templet = []
        for index in range(len(data_frame)):
            question = data_frame["question"][index]
            cypher = data_frame["cypher"][index]
            answer = data_frame["answer"][index]
            check = data_frame["check"][index]
            self.templet.append([question, cypher, answer, json.loads(check)])

    # 获取问题中包含的所有实体、关系、属性、标签
    def _get_contains_info(self, query) -> dict:
        # 问题中包含的所有实体
        contain_entities = re.findall("|".join(self.entities), query)
        # 问题中包含的所有关系
        contain_relations = re.findall("|".join(self.relations), query)
        # 问题中包含的所有属性
        contain_attributes = re.findall("|".join(self.attributes), query)
        # 问题中包含的所有标签
        contain_labels = re.findall("|".join(self.labels), query)
        return {
            "%ENT%": contain_entities,
            "%REL%": contain_relations,
            "%ATT%": contain_attributes,
            "%LAB%": contain_labels
        }

    # 根据模板中的check匹配模板
    def _get_match_templet(self, contains_info) -> dict:
        matched_templet = []
        for temp in self.templet:
            flag = True
            check = temp[3]
            for key, count in check.items():
                if len(contains_info.get(key,[])) < count:
                    flag = False
                    break
            if flag:
                matched_templet.append(temp)
        return matched_templet

    def _parse_value_pairs(self, value_pairs, check):
        params = {}
        for value_pair in value_pairs:
            for index, (key, count) in enumerate(check.items()):
                if count == 1:
                    params[key] = value_pair[index][0]
                else:
                    for i in range(count):
                        key_num = key[:-1] + str(i) + "%"
                        params[key_num] = value_pair[index][i]
        return params

    def _parse_templet(self, params, templet):
        question, cypher, answer = templet[0], templet[1], templet[2]
        for key, value in params.items():
            question = question.replace(key, value)
            answer = answer.replace(key, value)
            cypher = cypher.replace(key, value)
        return [question, cypher, answer]

    def _get_parsed_matched_templet(self, contains_info, matched_templet):
        parsed_matched_templet = []
        for temp in matched_templet:
            check = temp[3]
            comb_pairs = []
            for key, count in check.items():
                # 排列组合C(n,m)：从n个元素中抽取m个元素组成一组，列出所有可能的组合
                comb_pairs.append(itertools.combinations(contains_info[key], count))
            # 将不同类型的参数（实体、属性、关系、标签）进行随机组合
            value_pairs = itertools.product(*comb_pairs)
            # 对组合进行解析，用于替换模板中的变量
            params = self._parse_value_pairs(value_pairs, check)
            # 替换模板中question、cypher、answer的变量
            parsed_matched_templet.append(self._parse_templet(params, temp))
        return parsed_matched_templet

    def _calc_similarity(self, parsed_matched_templet, query):
        for temp in parsed_matched_templet:
            question = temp[0]
            # 计算Jaccard相似度（两字符串的交集个数除以并集个数）
            jaccard_similarity = len(set(question) & set(query)) / len(set(question) | set(query))
            temp.append(jaccard_similarity)
        # 根据相似度从大到小排序
        return sorted(parsed_matched_templet, key=lambda x: x[-1], reverse=True)

    def query(self, query):
        # 获取问题中包含的实体、关系、属性、标签
        contains_info = self._get_contains_info(query)
        # 根据check匹配模板
        matched_templet = self._get_match_templet(contains_info)
        if len(matched_templet) == 0:
            print("该问题没有匹配的模板！！！")
            return
        # 解析模板中的变量，将变量替换成文本
        parsed_matched_templet = self._get_parsed_matched_templet(contains_info, matched_templet)
        # 计算原始输入和模板中的question相似度，根据相似度排序
        parsed_matched_templet = self._calc_similarity(parsed_matched_templet, query)
        for temp in parsed_matched_templet:
            cypher = temp[1]
            result = self.graph.run(cypher).data()
            if result:
                return list(result[0].values())[0]


if __name__ == '__main__':
    from config import Config

    graph = GraphQA(Config)

    answer = graph.query("王力宏的英文名是什么？")
    print(answer)

    answer = graph.query("《唯一》的发行年份是什么时候？")
    print(answer)

    answer = graph.query("无问西东的主演有谁？")
    print(answer)

    answer = graph.query("王力宏毕业院校是哪里？")
    print(answer)