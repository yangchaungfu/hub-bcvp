"""
保险公司Agent
"""

import os
import json
from openai import OpenAI
import insurance_api
import insurance_api_schema

# 函数名到api的映射
funcname2api = {
    "get_insurance_products": insurance_api.get_insurance_products,
    "get_product_detail": insurance_api.get_product_detail,
    "cal_cash_value": insurance_api.cal_cash_value,
    "compare_products": insurance_api.compare_products
}

# 大模型的key与url
llmname2url = {
    "qwen-plus": (os.getenv("DASHSCOPE_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "deepseek-chat": (os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com")
}

# 可以选其它模型试试
selected_model = "qwen-plus"

# 初始化openAI客户端
llm_client = OpenAI(
    api_key=llmname2url[selected_model][0],
    base_url=llmname2url[selected_model][1]
)


# 调用大模型
def call_llm(messages: list):
    response = llm_client.chat.completions.create(
        model=selected_model,
        messages=messages,
        tools=insurance_api_schema.api_schema,
        tool_choice="auto"  # 让模型自主决定是否调用工具
    )
    return response.choices[0].message


def run_agent(user_query: str):
    """
    运行Agent，处理用户查询
    Args:
        user_query: 用户输入的问题
        model: 使用的模型名称
    """

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的保险顾问助手。你可以：
                            1. 介绍各种保险产品及其详细信息
                            2. 根据客户需求计算保单的现金价值
                            3. 比较不同保险产品
                            请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。
                        """
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

    max_iterations = 8  # Agent循环：最多进行5轮工具调用
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        # 调用大模型
        response_msg = call_llm(messages)

        # 将模型响应加入对话历史
        messages.append(response_msg)

        # 检查是否需要调用工具
        tool_calls = response_msg.tool_calls
        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_msg.content)
            print("=" * 60)
            return response_msg.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行对应的函数
            if function_name in funcname2api.keys():
                function_to_call = funcname2api[function_name]
                function_return = function_to_call(**function_args)
                print(f"工具返回: {function_return[:200]}..." if len(
                    function_return) > 200 else f"工具返回: {function_return}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_return
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"


if __name__ == "__main__":
    user_querys = [
        "你们有哪些保险产品？",
        "我想了解一下人寿保险的详细信息",
        "我想知道「长相伴」（精英版）在各个年度的现金价值，假如我投保金额是20万的话。",
        "如果我花20万买份增额终身寿险，请问5年或10年后，我退保能获得的现金价值是多少？",
        "帮我比较一下所有终身寿险的产品，假设投保金额都是20万、投保年数是10年。"
    ]

    for i, user_query in enumerate(user_querys, 1):
        # print(f"{i}. {user_query}")
        run_agent(user_query)
