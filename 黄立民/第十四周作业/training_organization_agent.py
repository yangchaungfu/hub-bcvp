"""
保险公司Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义

def get_school_intro():
    """
    获取所有可用的保险产品列表
    """
    products = [
            {
                "id": 1,
                "name": "教学设施",
                "content": "配备两栋专业化教学楼，教室宽敞明亮，均配备多媒体教学设备及对应课程专用教具，分学科打造沉浸式教学空间，保障课堂教学质量。"
            },
            {
                "id": 2,
                "name": "图书馆",
                "content": "校内设有独立图书馆，馆藏各类少儿读物、艺术典籍、科普书籍，满足学员课后阅读与知识拓展需求。"
            },
            {
                "id": 3,
                "name": "食堂",
                "content": "标准化食堂每日提供营养均衡的餐食，严格把控食材安全与膳食搭配。"
            },
            {
                "id": 4,
                "name": "宿舍",
                "content": "温馨舒适的学生宿舍，配备专人管理，为寄宿学员提供安全便捷的住宿服务，培养独立生活能力。"
            }
        ]
    return json.dumps(products, ensure_ascii=False)


def get_course_detail(product_id: str):
    """
    获取指定课程的详细信息

    Args:
        product_id: 课程ID
    """
    products = {
        "art_001": {
            "id": "art_001",
            "name": "美术课",
            "type": "素质课程",
            "content": "涵盖素描、色彩、创意绘画等内容，注重培养孩子的观察力、想象力和审美能力，通过多元化创作形式，释放艺术天赋，提升动手创作能力。"
        },
        "music_001": {
            "id": "music_001",
            "name": "音乐课",
            "type": "素质课程",
            "content": "包含乐理知识、声乐训练、乐器认知等模块，采用趣味教学法激发对音乐的热爱，帮助掌握基础音乐技能，提升乐感与艺术表现力。"
        },
        "program_001": {
            "id": "program_001",
            "name": "编程课",
            "type": "科技课程",
            "content": "聚焦少儿编程基础，通过图形化编程、逻辑思维训练等内容，培养逻辑推理、问题解决能力，搭建科技思维框架，适应未来发展需求。"
        },
        "handwriting_001": {
            "id": "handwriting_001",
            "name": "书法课",
            "type": "传统文化课程",
            "content": "兼顾硬笔与软笔书法，从笔画、结构入手，规范书写习惯，传承中华传统文化，在笔墨书香中提升专注力与文化素养。"
        },
        "piano_001": {
            "id": "piano_001",
            "name": "钢琴课",
            "type": "器乐课程",
            "content": "一对一精准教学，从基础指法、识谱训练到曲目演奏，循序渐进提升钢琴演奏技能，培养音乐素养与舞台表现力，挖掘器乐演奏潜力。"
        }
    }

    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "课程不存在"}, ensure_ascii=False)


def get_course_fee(product_id: str, insured_amount: int, years: int, age: int):
    """
    计算保费

    Args:
        product_id: 产品ID
        insured_amount: 投保金额（元）
        years: 保障年限
        age: 投保人年龄
    """
    # 简化的保费计算逻辑（实际会更复杂）

    base_rates = {
        "art_001": 20000,
        "music_001": 20000,
        "program_001": 15000,
        "handwriting_001": 10000,
        "piano_001": 10000,

    }

    if product_id not in base_rates:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)

    base_rate = base_rates[product_id]

    # 年龄系数（年龄越大，费用越少）
    age_factor = abs(age - 18)

    # 计算年保费
    # annual_premium = insured_amount * base_rate * age_factor * year_factor
    total_cost = base_rate + age_factor * 1000

    result = {
        "product_id": product_id,
        "age": age,
        # "annual_premium": round(annual_premium, 2),
        "total_cost": total_cost,
        "calculation_note": f"基于{age}岁，报名{product_id}，学费{total_cost}元"
    }

    return json.dumps(result, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_school_intro",
            "description": "获取学校的基本情况，有哪些基础设置，例如图书馆等",
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
            "name": "get_course_detail",
            "description": "获取指定课程的详细信息，包括类型、具体内容等",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID，例如：art_001, music_001, program_001"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_course_fee",
            "description": "计算课程所需费用",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID"
                    },
                    "total_cost": {
                        "type": "integer",
                        "description": "报名课程所需费用"
                    },

                    "age": {
                        "type": "integer",
                        "description": "购买人年龄"
                    }
                },
                "required": ["product_id", "insured_amount", "years", "age"]
            }
        }
    },


]

# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_school_intro": get_school_intro,
    "get_course_detail": get_course_detail,
    "get_course_fee": get_course_fee
}


# model_name="deepseek-chat",  # DeepSeek 对话模型名称
def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行Agent，处理用户查询

    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key="sk-c5c3d5999bd940df920318af3d98ec4d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的某个培训机构的招生老师。你可以：
            1. 介绍各种学校的基础设施
            2. 介绍学习各种课程
            3. 计算报名课程的费用

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
        # print(response)

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


# ==================== 示例场景 ====================

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#" * 60)
    print("# 保险公司Agent演示 - Function Call能力展示")
    print("#" * 60)

    # 注意：需要设置环境变量 DASHSCOPE_API_KEY
    # 或者在调用时传入api_key参数

    scenarios = [
        "你们有哪些基础设施？",
        "我想了解一下音乐课的详细信息",
        "我女儿今年10岁，想学书法课需要多少钱？"
    ]

    print("\n以下是几个示例场景，您可以选择其中一个运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")

    print("\n" + "-" * 60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-" * 60)


if __name__ == "__main__":
    # 展示示例场景
    # demo_scenarios()

    # 运行示例（取消注释下面的代码来运行）
    # 注意：需要先设置环境变量 DASHSCOPE_API_KEY

    # 示例1：查询产品列表
    run_agent("你们有哪些基础设施？", model="qwen-plus")
    run_agent("我12岁，想报名学艺术课，需要多少钱？", model="qwen-plus")

    # 示例2：查询产品详情
    # run_agent("我想了解一下人寿保险的详细信息", model="qwen-plus")

    # 示例3：计算保费
    # run_agent("我今年35岁，想买50万保额的人寿保险，保20年，需要多少钱？", model="qwen-plus")

    # 示例4：计算收益
    # run_agent("如果我投保100万的人寿保险30年，到期能有多少收益？", model="qwen-plus")

    # 示例5：比较产品
    # run_agent("帮我比较一下人寿保险和意外险，保额都是100万，我35岁，保20年", model="qwen-plus")

    # 自定义查询
    # run_agent("你的问题", model="qwen-plus")


