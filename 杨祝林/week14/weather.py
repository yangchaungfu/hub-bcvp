import requests
import json
from typing import Dict, List, Any
import json5

# ===================== 配置区（重点修改：替换为你的大模型接口/密钥）=====================
# 示例：可替换为【即梦AI】/GPT-3.5/4o/通义千问/文心一言的Function Call接口
LLM_CONFIG = {
    "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  # 你的大模型接口地址
    "api_key": "sk-f38680a87e944aca94a3a0e46c991de9",  # 你的大模型密钥
    "model": "qwen-plus",  # 模型名称（如即梦AI的模型名、gpt-3.5-turbo）
    "temperature": 0.0  # Function Call建议设为0，保证结果确定性
}

# ===================== 1. 自定义工具函数（核心：你自己的业务逻辑，可无限扩展）=====================
def get_weather(city: str, date: str = "today") -> str:
    """
    天气查询函数：获取指定城市指定日期的天气
    :param city: 城市名（必选，如北京、上海）
    :param date: 日期（可选，默认today，如2026-01-30）
    :return: 自然语言的天气结果
    """
    # 实际场景中可替换为真实天气API（如高德、百度天气API）
    mock_weather = {
        "北京-today": "晴，气温-5~8℃，北风2级，空气质量优",
        "上海-today": "多云，气温3~10℃，东风1级，空气质量良",
        "广州-2026-01-30": "小雨，气温15~20℃，南风3级，空气质量优"
    }
    key = f"{city}-{date}"
    return mock_weather.get(key, f"未查询到{city}{date}的天气信息")

def calculator(expression: str) -> str:
    """
    计算器函数：执行简单的数学表达式计算（加减乘除、括号）
    :param expression: 数学表达式（必选，如1+2*3、(5-2)/3）
    :return: 计算结果，异常时返回错误信息
    """
    try:
        # 仅用于示例，生产环境需做严格的表达式校验，防止注入
        result = eval(expression, {"__builtins__": None}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败，错误原因：{str(e)}（仅支持加减乘除和括号的简单表达式）"

# 函数注册表：将【函数名】与【实际函数对象】映射，方便后续调用
FUNCTION_REGISTRY = {
    "get_weather": get_weather,
    "calculator": calculator
}

# ===================== 2. 定义函数元描述（大模型识别用，严格按大模型规范）=====================
# 格式说明：name=函数名（与注册表一致），description=函数用途（大模型据此判断是否调用），
# parameters=入参定义（type=object，properties=参数字段+类型+描述，required=必选参数）
FUNCTION_DESCRIPTIONS: List[Dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "用于查询指定城市在指定日期的天气情况，当用户询问天气相关问题时调用",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "要查询的城市名称，如北京、上海、广州"
                },
                "date": {
                    "type": "string",
                    "description": "要查询的日期，可选，默认是今天，格式如today或2026-01-30"
                }
            },
            "required": ["city"]  # city是必选参数，date可选
        }
    },
    {
        "name": "calculator",
        "description": "用于执行简单的数学表达式计算，支持加减乘除和括号，当用户需要计算数学问题时调用",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "需要计算的数学表达式，如1+2*3、(5-2)/3"
                }
            },
            "required": ["expression"]  # expression是必选参数
        }
    }
]

# ===================== 3. 大模型请求函数：获取大模型的Function Call指令=====================
def call_llm(messages: List[Dict[str, str]], need_function_call: bool = True) -> str:
    """
    调用大模型接口，获取响应内容
    :param messages: 大模型的对话消息体（[{"role": "user/assistant/system", "content": "..."}]）
    :param need_function_call: 是否需要大模型进行函数调用（True则传入函数描述）
    :return: 大模型的纯文本响应
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_CONFIG['api_key']}"  # 多数大模型的授权格式，可按需修改
    }
    payload = {
        "model": LLM_CONFIG["model"],
        "messages": messages,
        "temperature": LLM_CONFIG["temperature"]
    }
    # 如果需要函数调用，将函数描述传入payload（主流大模型均支持functions字段）
    if need_function_call:
        payload["functions"] = FUNCTION_DESCRIPTIONS
        # 强制大模型调用函数（可选：auto=大模型自主判断，none=不调用，function=强制调用）
        payload["function_call"] = "auto"

    try:
        response = requests.post(LLM_CONFIG["url"], headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # 抛出HTTP请求错误
        res_data = response.json()
        # 提取大模型响应（主流大模型的响应均在choices[0].message.content/function_call）
        message = res_data["choices"][0]["message"]
        # 如果大模型返回了函数调用指令，返回指令的JSON字符串；否则返回自然语言内容
        if "function_call" in message and message["function_call"] is not None:
            return json.dumps(message["function_call"], ensure_ascii=False)
        else:
            return message.get("content", "大模型未返回有效内容")
    except Exception as e:
        return f"大模型调用失败：{str(e)}"

# ===================== 4. 函数调用执行器：解析大模型指令并执行自定义函数=====================
def execute_function(function_call_str: str) -> str:
    """
    解析大模型的函数调用指令，执行对应的自定义函数
    修复点：处理大模型返回arguments为JSON字符串的情况，新增二次解析逻辑
    :param function_call_str: 大模型返回的函数调用指令（JSON字符串：{"name": "...", "arguments": {.../""{...}""}}）
    :return: 函数执行结果（失败则返回错误信息）
    """
    try:
        # 第一步：解析大模型的整体函数调用指令（用json5兼容非严格JSON）
        function_call = json5.loads(function_call_str)
        function_name = function_call.get("name")
        function_kwargs = function_call.get("arguments", {})

        # 核心修复：判断arguments是否是JSON字符串，若是则二次解析为字典
        if isinstance(function_kwargs, str):
            try:
                function_kwargs = json5.loads(function_kwargs)  # 二次解析字符串为字典
                print(f"检测到arguments为JSON字符串，已二次解析为字典：{function_kwargs}")
            except Exception as e:
                return f"arguments二次解析失败：{str(e)}（原始字符串：{function_kwargs}）"

        # 校验函数是否在注册表中
        if function_name not in FUNCTION_REGISTRY:
            return f"错误：未找到注册的函数「{function_name}」"
        # 获取实际函数对象并执行
        target_func = FUNCTION_REGISTRY[function_name]
        result = target_func(** function_kwargs)
        return f"函数「{function_name}」调用成功，结果：{result}"
    except Exception as e:
        return f"函数执行失败：{str(e)}（指令：{function_call_str}）"

# ===================== 5. 主流程：整合所有环节，完成Function Call全链路=====================
def function_call_agent(user_query: str) -> str:
    """
    Function Call主代理：处理用户查询，完成「大模型识别→函数调用→结果整合→生成回答」
    :param user_query: 用户的原始问题
    :return: 最终的自然语言回答
    """
    # 第一步：向大模型发送用户问题，获取函数调用指令（或直接自然语言回答）
    print(f"用户问题：{user_query}")
    messages = [{"role": "user", "content": user_query}]
    llm_response = call_llm(messages, need_function_call=True)
    print(f"大模型初步响应：{llm_response}")

    # 第二步：判断大模型是否返回了函数调用指令（是否为JSON格式的{name, arguments}）
    try:
        # 尝试解析为函数调用指令，能解析则执行函数
        function_result = execute_function(llm_response)
        print(f"函数执行结果：{function_result}")

        # 第三步：将函数执行结果回传给大模型，让大模型生成自然语言最终回答
        messages.append({"role": "assistant", "content": llm_response})  # 大模型的函数调用指令
        messages.append({"role": "user", "content": f"请根据以下函数执行结果，用自然语言回答我的问题：{function_result}"})
        final_answer = call_llm(messages, need_function_call=False)  # 无需再调用函数
        return final_answer
    except:
        # 若不是函数调用指令，直接返回大模型的自然语言回答
        return llm_response

# ===================== 测试运行=====================
if __name__ == "__main__":
    # 测试用例1：天气查询（触发get_weather函数）
    print("===== 测试用例1：天气查询 =====")
    print(function_call_agent("请问今天北京的天气怎么样？"))
    print("\n" + "-"*50 + "\n")

    # 测试用例2：数学计算（触发calculator函数）
    print("===== 测试用例2：数学计算 =====")
    print(function_call_agent("帮我计算一下(100-20)*5+18等于多少？"))
    print("\n" + "-"*50 + "\n")

    # 测试用例3：无需函数调用（大模型直接回答）
#    print("===== 测试用例3：无需函数调用 =====")
#    print(function_call_agent("请解释一下什么是Function Call？"))
