import json

def load_data(data_path):
    titles = []
    contents = []
    with open(data_path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            titles.append(title)
            contents.append(content)
    return titles, contents

def average_string_length(strings_list):
    if not strings_list:  # 处理空列表的情况
        return 0
    total_length = sum(len(s) for s in strings_list)
    return total_length / len(strings_list)

if __name__ == "__main__":
    titles, contents = load_data(r"sample_data.json")
    print("标题的平均长度：", average_string_length(titles))   # 19.60576923076923
    print("内容平均长度：", average_string_length(contents))   # 105.04807692307692
