"""
证券公司Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 每个证券公司有自己不同的产品，需要企业自己定义

def get_securities_products():
    """
    获取所有可用的证券产品列表
    """
    products = [
        {
            "id": "stock_001",
            "name": "蓝筹股票组合",
            "type": "股票",
            "description": "投资于大型优质上市公司股票",
            "min_amount": 10000,
            "max_amount": 5000000,
            "risk_level": "中高",
            "expected_return": "8-15%"
        },
        {
            "id": "bond_001",
            "name": "国债逆回购",
            "type": "债券",
            "description": "短期低风险理财工具",
            "min_amount": 1000,
            "max_amount": 10000000,
            "risk_level": "低",
            "expected_return": "2-4%"
        },
        {
            "id": "fund_001",
            "name": "混合型基金",
            "type": "基金",
            "description": "股债混合配置，平衡风险与收益",
            "min_amount": 1000,
            "max_amount": 5000000,
            "risk_level": "中",
            "expected_return": "5-10%"
        },
        {
            "id": "etf_001",
            "name": "指数ETF",
            "type": "ETF",
            "description": "跟踪主要指数，分散投资风险",
            "min_amount": 100,
            "max_amount": 10000000,
            "risk_level": "中",
            "expected_return": "6-12%"
        }
    ]
    return json.dumps(products, ensure_ascii=False)


def get_product_detail(product_id: str):
    """
    获取指定证券产品的详细信息

    Args:
        product_id: 产品ID
    """
    products = {
        "stock_001": {
            "id": "stock_001",
            "name": "蓝筹股票组合",
            "type": "股票",
            "description": "投资于大型优质上市公司股票，如银行、保险、能源等行业的龙头企业",
            "features": ["流动性好", "分红稳定", "长期增值潜力大"],
            "min_amount": 10000,
            "max_amount": 5000000,
            "risk_level": "中高",
            "expected_return": "8-15%",
            "fees": {
                "commission": "0.025%",    # 佣金
                "stamp_duty": "0.1%（卖出时）",  # 印花税
                "management_fee": "无"     # 管理费
            }
        },
        "bond_001": {
            "id": "bond_001",
            "name": "国债逆回购",
            "type": "债券",
            "description": "以国债作抵押的短期贷款，安全性高",
            "features": ["安全性高", "期限灵活", "收益稳定"],
            "min_amount": 1000,
            "max_amount": 10000000,
            "risk_level": "低",
            "expected_return": "2-4%",
            "fees": {
                "commission": "0.001%",
                "stamp_duty": "无",
                "management_fee": "无"
            }
        },
        "fund_001": {
            "id": "fund_001",
            "name": "混合型基金",
            "type": "基金",
            "description": "同时投资于股票和债券，根据市场情况动态调整比例",
            "features": ["专业管理", "分散风险", "灵活配置"],
            "min_amount": 1000,
            "max_amount": 5000000,
            "risk_level": "中",
            "expected_return": "5-10%",
            "fees": {
                "commission": "无",
                "stamp_duty": "无",
                "management_fee": "1.5%/年",
                "subscription_fee": "1.0%",
                "redemption_fee": "0.5%（持有不足1年）"
            }
        },
        "etf_001": {
            "id": "etf_001",
            "name": "指数ETF",
            "type": "ETF",
            "description": "跟踪沪深300等主要指数，实现指数化投资",
            "features": ["分散投资", "透明度高", "交易灵活"],
            "min_amount": 100,
            "max_amount": 10000000,
            "risk_level": "中",
            "expected_return": "6-12%",
            "fees": {
                "commission": "0.025%",
                "stamp_duty": "无",
                "management_fee": "0.5%/年"
            }
        }
    }

    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)


def calculate_return(product_id: str, investment_amount: float, years: int):
    """
    计算投资收益

    Args:
        product_id: 产品ID
        investment_amount: 投资金额（元）
        years: 投资年限
    """
    # 不同产品的预期年化收益率
    expected_returns = {
        "stock_001": 0.10,  # 股票预期年化收益率
        "bond_001": 0.03,  # 债券预期年化收益率
        "fund_001": 0.07,  # 基金预期年化收益率
        "etf_001": 0.08  # ETF预期年化收益率
    }

    if product_id not in expected_returns:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)

    annual_rate = expected_returns[product_id]

    # 复利计算
    total_value = investment_amount * ((1 + annual_rate) ** years)
    total_return = total_value - investment_amount

    # 计算费用
    fees = 0
    if product_id == "stock_001":
        fees = investment_amount * 0.00025 + total_value * 0.001  # 佣金+印花税
    elif product_id == "bond_001":
        fees = investment_amount * 0.00001  # 佣金
    elif product_id == "fund_001":
        fees = investment_amount * 0.01 + investment_amount * 0.015 * years  # 申购费+管理费
    elif product_id == "etf_001":
        fees = investment_amount * 0.00025 + investment_amount * 0.005 * years  # 佣金+管理费

    net_return = total_return - fees

    result = {
        "product_id": product_id,
        "investment_amount": investment_amount,
        "years": years,
        "annual_return_rate": f"{annual_rate * 100}%",
        "maturity_value": round(total_value, 2),
        "total_return": round(total_return, 2),
        "total_fees": round(fees, 2),
        "net_return": round(net_return, 2),
        "note": "此为基于历史数据的预期收益，实际收益可能有所不同"
    }

    return json.dumps(result, ensure_ascii=False)


def analyze_risk(product_id: str):
    """
    分析产品风险

    Args:
        product_id: 产品ID
    """
    risk_analysis = {
        "stock_001": {
            "risk_level": "中高",
            "risk_factors": [
                "市场风险：股票价格受市场整体波动影响",
                "行业风险：特定行业可能面临政策或周期性风险",
                "个股风险：个别公司可能面临经营风险"
            ],
            "risk_mitigation": [
                "分散投资于多个行业和公司",
                "长期持有以平滑短期波动",
                "定期调整投资组合"
            ]
        },
        "bond_001": {
            "risk_level": "低",
            "risk_factors": [
                "利率风险：市场利率上升可能导致债券价格下跌",
                "流动性风险：极端市场情况下可能难以快速变现"
            ],
            "risk_mitigation": [
                "选择短期品种降低利率风险",
                "分散投资于不同期限的债券"
            ]
        },
        "fund_001": {
            "risk_level": "中",
            "risk_factors": [
                "市场风险：受股票和债券市场双重影响",
                "基金管理风险：基金经理的投资决策可能影响业绩",
                "流动性风险：大额赎回可能影响基金运作"
            ],
            "risk_mitigation": [
                "选择历史业绩稳定的基金",
                "分散投资于不同类型的基金",
                "定期关注基金持仓和业绩表现"
            ]
        },
        "etf_001": {
            "risk_level": "中",
            "risk_factors": [
                "市场风险：跟踪指数的波动直接影响ETF价格",
                "跟踪误差：ETF可能无法完全复制指数表现"
            ],
            "risk_mitigation": [
                "选择流动性好、规模大的ETF",
                "分散投资于不同类型的ETF",
                "长期持有以降低交易成本"
            ]
        }
    }

    if product_id in risk_analysis:
        return json.dumps(risk_analysis[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)


def recommend_products(risk_preference: str, investment_amount: float, investment_period: str):
    """
    根据用户风险偏好和投资需求推荐产品

    Args:
        risk_preference: 风险偏好（低、中、高）
        investment_amount: 投资金额（元）
        investment_period: 投资期限（短期、中期、长期）
    """
    # 获取所有产品
    all_products = json.loads(get_securities_products())

    # 根据风险偏好筛选产品
    risk_map = {
        "低": ["低"],
        "中": ["低", "中"],
        "高": ["低", "中", "中高"]
    }

    acceptable_risks = risk_map.get(risk_preference, ["中"])
    filtered_products = [p for p in all_products if p["risk_level"] in acceptable_risks]

    # 根据投资金额筛选产品
    filtered_products = [p for p in filtered_products
                         if p["min_amount"] <= investment_amount <= p["max_amount"]]

    # 根据投资期限筛选产品（简化处理）
    period_map = {
        "短期": ["bond_001"],
        "中期": ["bond_001", "fund_001", "etf_001"],
        "长期": ["stock_001", "fund_001", "etf_001"]
    }

    suitable_products = period_map.get(investment_period, [])
    filtered_products = [p for p in filtered_products if p["id"] in suitable_products]

    # 如果没有匹配的产品，则返回所有符合风险偏好的产品
    if not filtered_products:
        filtered_products = [p for p in all_products if p["risk_level"] in acceptable_risks]

    # 计算每个产品的预期收益
    for product in filtered_products:
        years = 1 if investment_period == "短期" else (3 if investment_period == "中期" else 5)
        return_result = json.loads(calculate_return(product["id"], investment_amount, years))
        product["expected_value"] = return_result.get("maturity_value", 0)
        product["expected_return"] = return_result.get("net_return", 0)

    # 按预期收益排序
    filtered_products.sort(key=lambda x: x["expected_return"], reverse=True)

    result = {
        "user_preference": {
            "risk_preference": risk_preference,
            "investment_amount": investment_amount,
            "investment_period": investment_period
        },
        "recommended_products": filtered_products
    }

    return json.dumps(result, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_securities_products",
            "description": "获取所有可用的证券产品列表，包括产品名称、类型、风险等级、预期收益率等基本信息",
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
            "description": "获取指定证券产品的详细信息，包括产品特点、费用结构等",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID，例如：stock_001, bond_001, fund_001, etf_001"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_return",
            "description": "计算指定证券产品的投资收益，包括总收益和净收益（扣除费用后）",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID"
                    },
                    "investment_amount": {
                        "type": "number",
                        "description": "投资金额（元）"
                    },
                    "years": {
                        "type": "integer",
                        "description": "投资年限（年）"
                    }
                },
                "required": ["product_id", "investment_amount", "years"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_risk",
            "description": "分析指定证券产品的风险因素和风险缓解措施",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": "根据用户的风险偏好、投资金额和投资期限推荐合适的证券产品",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_preference": {
                        "type": "string",
                        "description": "风险偏好（低、中、高）"
                    },
                    "investment_amount": {
                        "type": "number",
                        "description": "投资金额（元）"
                    },
                    "investment_period": {
                        "type": "string",
                        "description": "投资期限（短期、中期、长期）"
                    }
                },
                "required": ["risk_preference", "investment_amount", "investment_period"]
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_securities_products": get_securities_products,
    "get_product_detail": get_product_detail,
    "calculate_return": calculate_return,
    "analyze_risk": analyze_risk,
    "recommend_products": recommend_products
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
            "content": """你是一位专业的证券投资顾问助手。你可以：
1. 介绍各种证券产品及其详细信息
2. 根据客户需求计算投资收益
3. 分析证券产品的风险因素
4. 根据客户的风险偏好和投资需求推荐合适的产品

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


# ==================== 示例场景 ====================

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#" * 60)
    print("# 证券公司Agent演示 - Function Call能力展示")
    print("#" * 60)

    # 注意：需要设置环境变量 DASHSCOPE_API_KEY
    # 或者在调用时传入api_key参数

    scenarios = [
        "你们有哪些证券产品？",
        "我想了解一下蓝筹股票组合的详细信息",
        "我投资10万元购买指数ETF，持有3年能有多少收益？",
        "股票投资有哪些风险？",
        "我风险偏好中等，有50万元想投资3年，有什么推荐的产品？"
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
    # run_agent("你们有哪些证券产品？", model="qwen-plus")

    # 示例2：查询产品详情
    # run_agent("我想了解一下蓝筹股票组合的详细信息", model="qwen-plus")

    # 示例3：计算收益
    # run_agent("我投资10万元购买指数ETF，持有3年能有多少收益？", model="qwen-plus")

    # 示例4：分析风险
    # run_agent("股票投资有哪些风险？", model="qwen-plus")

    # 示例5：推荐产品
    run_agent("我风险偏好中等，有50万元想投资3年，有什么推荐的产品？", model="qwen-plus")

    # 自定义查询
    # run_agent("你的问题", model="qwen-plus")
