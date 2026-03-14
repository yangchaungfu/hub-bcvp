"""
保险公司Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from zai import ZhipuAiClient


# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义

def get_cold_types():
    """
    获取所有感冒类型
    """
    products = [
        {
            "id": "cold_001",
            "name": "风热感冒",
            "type": "cold",
            "description": "风热感冒"
        },
        {
            "id": "cold_002",
            "name": "风寒感冒",
            "type": "cold",
            "description": "风寒感冒"
        },
        {
            "id": "cold_003",
            "name": "流感",
            "type": "flu",
            "description": "流感"
        }
    ]
    return json.dumps(products, ensure_ascii=False)


def get_cold_detail(cold_id: str):
    """
    获取感冒具体描述

    Args:
        cold_id: 感冒类型ID
    """
    colds = {
        "cold_001": {
            "id": "cold_001",
            "name": "风热感冒",
            "type": "cold",
            "main_description": "发热重、怕冷轻，有汗，以呼吸道热症、内热表现为核心，多因气温偏高、内热上火后吹风引发。",
            "specific_symptoms": "鼻塞流黄稠涕，咳嗽吐黄稠痰；咽喉肿痛明显（最典型表现），口干、口苦、总想喝水；头痛、头晕，舌尖发红，舌苔薄黄，脉象浮数；部分人会有轻微出汗、面红目赤的表现。"
        },
        "cold_002": {
            "id": "cold_002",
            "name": "风寒感冒",
            "type": "cold",
            "main_description": "怕冷重、发热轻，无汗，以呼吸道寒症、全身冷痛为主要表现，多因吹风受凉、接触寒凉引发。",
            "specific_symptoms": "鼻塞流清涕，咳嗽吐白稀痰；头痛、全身肌肉关节酸痛，四肢发寒；无口干、咽喉肿痛，舌苔薄白，脉象浮紧；部分人会有轻微怕冷、打寒颤的表现。"
        },
        "cold_003": {
            "id": "cold_003",
            "name": "流感",
            "type": "flu",
            "main_description": "传染性强，突发重症，与普通风热 / 风寒感冒的核心区别是全身症状远重于局部呼吸道症状，多在秋冬春季节爆发，易引发聚集性感染。",
            "specific_symptoms": "突发高烧（39~40℃），伴随明显畏寒、寒战；全身肌肉酸痛、关节剧痛，头痛剧烈，全身乏力、疲倦感显著（甚至卧床不起）；局部呼吸道症状较轻，可能伴随轻微咳嗽、咽痛、流涕（清涕 / 黄涕均可能），部分人会有口干、口苦；少数人会出现恶心、呕吐、腹泻（胃肠型流感，儿童 / 老人更常见）；病程比普通感冒长，若未及时干预，易引发肺炎、支气管炎等并发症。"
        }
    }

    if cold_id in colds:
        return json.dumps(colds[cold_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "没有对应的感冒症状描述"}, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_cold_types",
            "description": "获取所有感冒类型列表，包括名称、类型等基本信息",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cold_detail",
            "description": "获取指定感冒类型的具体症状及核心特点",
            "parameters": {
                "type": "object",
                "properties": {
                    "cold_id": {
                        "type": "string",
                        "description": "cold_id 感冒类型ID， 例如 cold_001"
                    }
                },
                "required": ["cold_id"]
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_cold_types": get_cold_types,
    "get_cold_detail": get_cold_detail
}


def run_agent(user_query: str, api_key: str = None, model: str = "glm-3-turbo"):
    """
    运行Agent，处理用户查询

    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = ZhipuAiClient(api_key=os.environ.get("zhipuApiKey")) #key填入环境变量

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的医疗顾问助手。你可以：
1. 介绍各种感冒类型及其详细信息
2. 可以描述对应感冒得到具体症状

请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "=" * 60)
    print("【用户问题】")
    print(user_query)
    print("=" * 60)

    # Agent循环：最多进行5轮工具调用
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )

        response_message = response.choices[0].message

        # 将模型响应加入对话历史
        messages.append(response_message)

        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"工具返回: {function_response[:200]}..." if len(
                    function_response) > 200 else f"工具返回: {function_response}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"


if __name__ == "__main__":
    # 展示示例场景
    # demo_scenarios()

    # 运行示例（取消注释下面的代码来运行）
    # 注意：需要先设置环境变量 DASHSCOPE_API_KEY

    # 示例1：查询产品列表
    # run_agent("我好像有点感冒了，但我不知道我是哪种感冒，你可以帮我吗？", model="glm-3-turbo")

    # 示例2：查询产品详情
    # run_agent("我有点咳嗽和鼻塞，我是哪种感冒？", model="glm-3-turbo")

    # 示例3：计算保费
    run_agent("风热感冒有哪些具体症状？", model="glm-3-turbo")
