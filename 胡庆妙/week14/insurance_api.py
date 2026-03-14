"""
保险咨询服务的接口。
"""

import json
from collections import defaultdict


def get_insurance_products():
    """
    获取所有可用的保险产品列表
    """
    products = [
        {
            "id": "life_001",
            "name": "「长相伴」（精英版）终身寿险",
            "type": "增额终身寿险",
            "company": "中国太平洋人寿保险股份有限公司",
            "issue_age": "0-70周岁",  # 投保年龄
            "coverage_years": "终身",  # 保障年限
            "payment_mode": "趸交（一次性交清）、3年交",  # 缴费方式
            "min_invested_amt": "趸交不低于20万；3年交，则每年不低于6.7万；5年交，则每年不低于4万；10年交，则每年不低于2万"
        },
        {
            "id": "life_002",
            "name": "「长相伴」（白领版）终身寿险",
            "type": "增额终身寿险",
            "company": "中国太平洋人寿保险股份有限公司",
            "issue_age": "0-70周岁",  # 投保年龄
            "coverage_years": "终身",  # 保障年限
            "payment_mode": "趸交（一次性交清）、3年交、5年交、10年交",  # 缴费方式
            "min_invested_amt": "趸交不低于10万；3年交，则每年不低于3.3万；5年交，则每年不低于2万；10年交，则每年不低于1万"
        }
    ]
    return json.dumps(products, ensure_ascii=False)


def get_product_detail(product_id: str):
    """
    获取指定保险产品的详细信息

    Args:
        product_id: 产品id
    """
    products = {
        "life_001": {
            "id": "life_001",
            "name": "「长相伴」（精英版）终身寿险",
            "type": "增额终身寿险",
            "company": "中国太平洋人寿保险股份有限公司",
            "issue_age": "0-70周岁",  # 投保年龄
            "coverage_years": "终身",  # 保障年限
            "payment_mode": "趸交（一次性交清）、3年交",  # 缴费方式
            "min_invested_amt": "趸交不低于20万；3年交，则每年不低于6.7万；5年交，则每年不低于4万；10年交，则每年不低于2万"
        },
        "life_002": {
            "id": "life_002",
            "name": "「长相伴」（白领版）终身寿险",
            "type": "增额终身寿险",
            "company": "中国太平洋人寿保险股份有限公司",
            "issue_age": "0-70周岁",  # 投保年龄
            "coverage_years": "终身",  # 保障年限
            "payment_mode": "趸交（一次性交清）、3年交、5年交、10年交",  # 缴费方式
            "min_invested_amt": "趸交不低于10万；3年交，则每年不低于3.3万；5年交，则每年不低于2万；10年交，则每年不低于1万"
        }
    }

    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)


def cal_cash_value(product_id: str, invested_amt: int, invested_years: int = None):
    """
    计算保单的现金价值
    Args:
        product_id: 产品id
        invested_amt: 投保金额（元）
        invested_years: 投保年限
    """
    # 产品与内部收益率的映射
    product_irr = {"life_001": 0.038,
                   "life_002": 0.028}

    # 保额/现金价值比例
    coverage_rate = 1.04

    if invested_years is None:
        years_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70]
    else:
        years_list = [invested_years]

    cash_value_list = list(defaultdict())
    for years in years_list:
        item = {}
        cash_value = invested_amt * ((1 + product_irr[product_id]) ** years)  # 复利计算
        coverage_amt = cash_value * coverage_rate * (1 + years / 1000)  # 保额计算
        item["product_id"] = product_id
        item["invested_amt"] = invested_amt
        item["invested_years"] = years
        item["cash_value"] = cash_value
        item["coverage_amt"] = coverage_amt
        cash_value_list.append(item)

    return json.dumps(cash_value_list, ensure_ascii=False)


def compare_products(product_ids: list, invested_amt: int, invested_years: int):
    """
    对多个保单的现金价值、保额、缴费方式进行比较
    Args:
        product_id: 产品id
        invested_amt: 已投资金额（元）
        invested_years: 投保年限
    """
    comparisons = []

    for product_id in product_ids:
        result = json.loads(cal_cash_value(product_id, invested_amt, invested_years))[0]

        # 获取产品名称
        product_detail = json.loads(get_product_detail(product_id))
        result["product_name"] = product_detail.get("name", "未知产品")
        result["payment_mode"] = product_detail.get("payment_mode", "未知缴费方式")
        comparisons.append(result)

    comparisons.sort(key=lambda x: x["cash_value"], reverse=True)
    result = {
        "comparison_params": {
            "insured_amount": invested_amt,
            "years": invested_years,
        },
        "product_compare_detail": comparisons
    }
    return json.dumps(result, ensure_ascii=False)
