# -*- coding:utf-8 -*-


import re
import ast
import pandas as pd


class CSVHandler:
    def __init__(self, text_path, out_path):
        self.text_path = text_path
        self.out_path = out_path
        self.json_regular = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
        self.dict_list = []
        self.parseJson2Dict()

    # 如果文本中包含json，使用此方法可以将文本中的json写入到csv文件
    def parseJson2Dict(self):
        if self.text_path is None:
            return
        with open(self.text_path, 'r', encoding='utf-8') as f:
            for line in f:
                matches = re.findall(self.json_regular, line, re.S)
                if matches:
                    for json_str in matches:
                        try:
                            # ast.literal_eval直接解析文本，将其转化为可能的python对象
                            normalized_str = (json_str
                                              .replace('true', 'True')
                                              .replace('false', 'False'))
                            data_dict = ast.literal_eval(normalized_str)
                            self.dict_list.append(data_dict)
                        except Exception:
                            # 匹配下一个
                            continue

    # 将dict写入到csv
    def write2csvByDicts(self, dicts=None, simplify=False):
        if dicts is None:
            dicts = self.dict_list
        if not isinstance(dicts, list):
            dicts = [dicts]
        # 对字段进行精简，只将必要的字段写入csv
        if simplify:
            keys = ["model_name", "train_type", "matching_type", "concat_type", "margin", "hidden_size", "out_channels"]
            dicts = [{k: v for k, v in d.items() if k in keys} for d in dicts]
        df = pd.DataFrame(dicts)
        # header默认值就是True，会将key作为标题
        df.to_csv(self.out_path, index=False, header=True, encoding='utf-8')



if __name__ == '__main__':
    from config import Config

    text_path = Config["log_path"] + "/evaluator.log"
    out_path = "./text.csv"
    handler = CSVHandler(text_path, out_path)
    handler.write2csv()
