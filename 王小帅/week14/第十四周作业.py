"""
保险公司Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义

def get_insurance_products():
    """
    获取所有的班级信息列表
    """
    products = [
        {
            "id": "grade_001",
            "name": "一班",
            "type": "一年级",
            "description": "希望小学一年级一班群体",
            "min_amount": 0,
            "max_amount": 35,
            "boy_amount": 20,
            "gril_amount": 15,
            "avg_heigh": "125cm"
        },
        {
            "id": "grade_002",
            "name": "二班",
            "type": "一年级",
            "description": "希望小学一年级二班群体",
            "min_amount": 0,
            "max_amount": 38,
            "boy_amount": 18,
            "gril_amount": 20,
            "avg_heigh": "130cm"
        },
        {
            "id": "grade_003",
            "name": "三班",
            "type": "一年级",
            "description": "希望小学一年级三班群体",
            "min_amount": 0,
            "max_amount": 43,
            "boy_amount": 21,
            "gril_amount": 22,
            "avg_heigh": "120cm"
        }
    ]
    return json.dumps(products, ensure_ascii=False)


def get_product_detail(product_id: str):
    """
    获取指定班级的详细信息
    
    Args:
        product_id: 班级ID
    """
    products = {
        "grade_001": {
            "id": "grade_001",
            "name": "一班",
            "type": "一年级",
            "description": "希望小学一年级一班群体",
            "coverage": ["活泼可爱", "善良纯真", "阳光少年"],
            "min_amount": 0,
            "max_amount": 35,
            "boy_amount": 20,
            "gril_amount": 15,
            "avg_heigh": "125cm"
        },
        "grade_002": {
            "id": "grade_002",
            "name": "二班",
            "type": "一年级",
            "description": "希望小学一年级二班群体",
            "coverage": ["诚实友善", "稚气未脱", "虎头虎脑"],
            "min_amount": 0,
            "max_amount": 38,
            "boy_amount": 18,
            "gril_amount": 20,
            "avg_heigh": "130cm"
        },
        "grade_003": {
            "id": "grade_003",
            "name": "三班",
            "type": "一年级",
            "description": "希望小学一年级三班群体",
            "coverage": ["勇敢大方", "诚实坦率", "懂礼谦让"],
            "min_amount": 0,
            "max_amount": 43,
            "boy_amount": 21,
            "gril_amount": 22,
            "avg_heigh": "120cm"
        }
    }
    
    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "没有该班级，请正确输入班级"}, ensure_ascii=False)


def compare_products(product_ids: list, avg_heigh: str, boy_amount: int, gril_amount: int):
    """
    比较不同班级的身高差、人数差、男女比例
    
    Args:
        product_id: 班级ID
        avg_heigh: 平均身高
        boy_amount: 男生人数
        gril_amount: 女生人数
    """
    comparisons = []

    for product_id in product_ids:
        comparisons.append(json.loads(get_product_detail(product_id)))

    # 按年保费排序
    comparisons.sort(key=lambda x: x["avg_heigh"])
    result = {
        "comparison_params": {
            "avg_heigh": avg_heigh,
            "boy_amount": boy_amount,
            "gril_amount": gril_amount
        },
        "products": comparisons
    }

    return json.dumps(result, ensure_ascii=False)

# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_insurance_products",
            "description": "获取所有班级，包括年级、总人数、平均身高、男女生数量等基本信息",
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
            "name": "get_product_detail",
            "description": "获取指定班级详细信息，包括总人数、平均身高、男女生数量等",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "班级ID，例如：grade_001, grade_002, grade_003"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "比较多个班级的不同差异",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要比较的班级ID列表"
                    },
                    "avg_heigh": {
                        "type": "integer",
                        "description": "平均身高（厘米）"
                    },
                    "boy_amount": {
                        "type": "integer",
                        "description": "男生人数（个）"
                    },
                    "gril_amount": {
                        "type": "integer",
                        "description": "女生人数（个）"
                    }
                },
                "required": ["product_ids", "avg_heigh", "boy_amount", "gril_amount"]
            }
        }
    }
]



# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_insurance_products": get_insurance_products,
    "get_product_detail": get_product_detail,
    "compare_products": compare_products
}


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
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位小学老师。你可以：
1. 介绍一年级各个班级的详细信息
2. 介绍不同班级的差异

请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    print("\n" + "="*60)
    print("【用户问题】")
    print(user_query)
    print("="*60)
    
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
            print("="*60)
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
                
                print(f"工具返回: {function_response[:200]}..." if len(function_response) > 200 else f"工具返回: {function_response}")
                
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

if __name__ == "__main__":
    # 展示示例场景
    # demo_scenarios()
    
    # 运行示例（取消注释下面的代码来运行）
    # 注意：需要先设置环境变量 DASHSCOPE_API_KEY
    
    # 示例1：查询产品列表
    run_agent("你们有哪些班级？", model="qwen-plus")
    
    # 示例2：查询产品详情
    # run_agent("我想了解一下不同班级的差异信息", model="qwen-plus")


