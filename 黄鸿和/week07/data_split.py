import json
from sklearn.model_selection import train_test_split

def load_split_and_save(input_json_filepath, train_json_filepath, test_json_filepath, test_size, random_seed = 42):
	data = None
	try:
		with open(input_json_filepath, 'r', encoding='utf-8') as f:
			data = json.load(f)
	except FileNotFoundError:
		print('连接错误')
		return
	except json.JSONDecodeError:
		print('不是json')
	except Exception as e:
		print(f'读取错误：{e}')
	
	try:
		train_data, test_data = train_test_split(
			data,
			test_size=test_size,
			random_state=random_seed
		)
		print('ok')
	except Exception as e:
		print('划分错误：{e}')
	
	try:
		with open(train_json_filepath, mode='w', encoding='utf-8') as f:
			for item in train_data: # 遍历列表中的每个字典
				# 将每个字典转换成 JSON 字符串，后面加上换行符
				f.write(json.dumps(item, ensure_ascii=False) + '\n')
	except Exception as e:
		print(f'train数据集错误：{e}')
	
	try:
		with open(test_json_filepath, mode='w', encoding='utf-8') as f:
			for item in train_data: # 遍历列表中的每个字典
				# 将每个字典转换成 JSON 字符串，后面加上换行符
				f.write(json.dumps(item, ensure_ascii=False) + '\n')
	except Exception as e:
		print(f'test数据集错误：{e}')
	

if __name__ == '__main__':
	input_json_file = r'F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\data\comments.json'
	train_json_file = r'F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\data\comments_train.json'
	test_json_file = r'F:\八斗学院\第七周 文本分类\week7 文本分类问题\nn_pipline_week07\data\comments_test.json'
	load_split_and_save(
		input_json_filepath=input_json_file,
		train_json_filepath=train_json_file,
		test_json_filepath=test_json_file,
		test_size=0.2  # 20% 作为测试集
	)
	print('--- 数据划分完成 ---')