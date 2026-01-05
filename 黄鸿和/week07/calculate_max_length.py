import pandas as pd
import numpy as np

def calculate_max_length(file_path, column_name):
	df = pd.read_csv(file_path)
	df['text_length'] = df[column_name].astype(str).apply(len)
	p95_length = np.percentile(
		df['text_length'],
		95,
		interpolation = 'higher'
	)
	return int(p95_length)


file_path = r'F:\八斗学院\第七周 文本分类\week7 文本分类问题\文本分类练习.csv'
column_name = 'review'

result = calculate_max_length(file_path, column_name)

print(f'最终计算出95% 数量文本的最大文本长度为：{result}')