import csv
import json
import os

def csv_to_json(Input_csv_file, Output_json_file, column_name1 = 'label', column_name2 = 'review'):
	data_list = []
	MAX_REVIEW_LENGTH = 69
	try:
		with open(Input_csv_file, mode='r', encoding='utf-8') as csvfile:
			reader = csv.DictReader(csvfile)
			fieldnames = reader.fieldnames
			if not fieldnames or column_name1 not in fieldnames or column_name2 not in fieldnames:
				print('没有字段')
				return
			json_column_name1 = 'tag'
			json_column_name2 = 'review'
			for row in reader:
				if len(row[column_name2]) <= MAX_REVIEW_LENGTH:
					data_list.append({
						json_column_name1: int(row[column_name1]),
						json_column_name2: row[column_name2]
						})
		with open(Output_json_file, mode = 'w', encoding='utf-8') as jsonfile:
			json.dump(data_list, jsonfile, ensure_ascii=False, indent=1)
	except FileNotFoundError:
		print(f'错误：未找到文件请检查路径。')
	except Exception as e:
		print(f"发生了一个错误: {e}")



Input_csv_file = r'F:\八斗学院\第七周 文本分类\week7 文本分类问题\文本分类练习.csv'
Output_json_file = r'F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\data\comments.json'

csv_to_json(Input_csv_file, Output_json_file)
print('转换完成')
