"""
保险咨询服务的接口描述。 LLM需要参考它以生成“调用工具”的提示词。
"""

api_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_insurance_products",
            "description": "获取所有可用的保险产品列表，列表中的每项产品包括产品名称、类型、保险公司、投保年龄范围、保障年限、缴费方式、最低投保金额等信息",
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
            "description": "根据产品id获取对应保险产品的详细信息，包括产品名称、类型、保险公司、投保年龄范围、保障年限、缴费方式、最低投保金额等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品id，例如：life_001, life_002"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cal_cash_value",
            "description": "指定产品id、投保金额、投保年数，计算保单的现金价值。如果没有指定投保年数，则计算保单在各个年度的现金价值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品id"
                    },
                    "invested_amt": {
                        "type": "integer",
                        "description": "投保金额（元）"
                    },
                    "invested_years": {
                        "type": "integer",
                        "description": "投保年数（年）"
                    }
                },
                "required": ["product_id", "invested_amt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "比较多个保险产品在相同条件下的现金价值",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要比较的产品id列表"
                    },
                    "invested_amt": {
                        "type": "integer",
                        "description": "投保金额（元）"
                    },
                    "invested_years": {
                        "type": "integer",
                        "description": "保障年限（年）"
                    }
                },
                "required": ["product_ids", "invested_amt", "invested_years"]
            }
        }
    }
]
