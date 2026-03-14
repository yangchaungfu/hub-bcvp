## 20260301-week17-第十七周作业

### 作业内容和要求：在任务型对话中加入重听功能

## 优化内容

### 1. 退出功能

**功能描述**：当用户任意时刻输出"退出"或"quit"时，退出脚本。

**实现方法**：在主循环中添加退出检查逻辑，当用户输入"退出"、"quit"、"Quit"或"QUIT"时，打印"再见！"并退出循环。

### 2. 提供可选项功能

**功能描述**：当用户输入不满足未识别时，将可选项一起列出来提供给用户参考。

**实现方法**：在`nlg`方法中，当系统需要追问用户时，从`slot_info`中提取可选项，并添加到响应中。

### 3. 重新介绍商品功能（重点）

**功能描述**：当用户选好了款式和尺寸后，已经推荐了商品，后续所有的问答环节都可以再次输入【重新介绍下商品】时，将商品详细信息再播报一遍，同时继续当前问答。

**实现方法**：
1. **保存商品推荐节点**：在`dpo`方法中，当系统执行商品推荐操作时，保存当前节点作为`product_node`。
2. **保存商品推荐响应**：在`nlg`方法中，只保存商品推荐的响应到`last_response`。
3. **重新介绍商品**：当用户输入"重新介绍下商品"时，根据保存的`last_response`或`product_node`重新生成商品推荐信息。
4. **继续当前问答**：重新介绍商品后，继续当前的问答流程，添加当前需要填充的槽位提示。

**核心修改代码**：

1. **保存商品推荐节点**：
```python
def dpo(self, memory):
    if memory["require_slot"] is None:
        memory["policy"] = "reply"
        hit_node = memory["hit_node"]
        if hit_node:
            node_info = self.all_node_info[hit_node]
            # 保存当前节点作为商品推荐节点
            if "action" in node_info and "select 衣服" in "".join(node_info.get("action", [])):
                memory["product_node"] = hit_node
            memory["available_nodes"] = node_info.get("childnode", [])
    # 其他逻辑...
    return memory
```

2. **只保存商品推荐的响应**：
```python
def nlg(self, memory):
    # 检查是否需要重新介绍商品
    if memory.get('query', '').strip() == "重新介绍下商品":
        if 'last_response' in memory:
            memory["response"] = memory['last_response']
            # 重新介绍商品后，继续当前的问答流程
            if memory.get("require_slot"):
                slot = memory["require_slot"]
                if slot and slot in self.slot_info:
                    query = self.slot_info[slot]["query"]
                    # 添加可选项
                    values = self.slot_info[slot].get("values", "")
                    # 解析并添加可选项...
                    memory["response"] += "\n" + query
            return memory
        elif 'product_node' in memory:
            # 根据product_node重新生成商品推荐
            node = memory["product_node"]
            if node in self.all_node_info:
                node_info = self.all_node_info[node]
                memory["response"] = self.fill_in_slot(node_info["response"], memory)
                memory['last_response'] = memory["response"]
                # 继续当前的问答流程...
            return memory
    
    # 根据policy执行反问或回答
    if memory["policy"] == "reply":
        hit_node = memory["hit_node"]
        if hit_node:
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
            # 只保存商品推荐的响应
            if "action" in node_info and "select 衣服" in "".join(node_info.get("action", [])):
                memory['last_response'] = memory["response"]
    # 其他逻辑...
    return memory
```

## 测试结果

### 1. 退出功能测试

**输入**：
```
请输入：退出
```

**输出**：
```
再见！
```

### 2. 提供可选项功能测试

**输入**：
```
请输入：a
```

**输出**：
```
您想买长袖、短袖还是半截袖 可选：长袖、短袖、半截袖
```

### 3. 重新介绍商品功能测试

**输入**：
```
请输入：短袖
请输入：红
请输入：s
请输入：
请输入：重新介绍下商品
```

**输出**：
```
为您推荐这一款，s号，红色短袖，产品连接：xxx
您想分多少期，可以有3期，6期，9期，12期 可选：3、6、9、12
```

## 总结

通过以上优化，对话系统现在具备以下功能：

1. **退出功能**：用户可以随时输入"退出"或"quit"退出对话。
2. **提供可选项**：当用户输入不明确时，系统会提供可选项供用户参考。
3. **重听功能**：用户可以在任何时候输入"重新介绍下商品"来获取商品的详细信息，同时系统会继续当前的问答流程。

这些优化使得对话系统更加用户友好，能够更好地满足用户的需求