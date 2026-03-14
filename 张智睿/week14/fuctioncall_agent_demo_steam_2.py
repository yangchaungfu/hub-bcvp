"""
Steam游戏平台Agent示例 - 演示大语言模型的function call能力
保留原框架（tools schema + available_functions + run_agent循环）
场景：Steam游戏平台（本地模拟数据，可替换为真实API）
工具：8个
末尾：提供可用示例问题
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="DASHSCOPE_API_KEY.env", override=False)


# ==================== 模拟Steam数据集（10个热门游戏，中文名） ====================

STEAM_CATALOG = [
    {
        "app_id": 730,
        "name": "反恐精英2",
        "genres": ["射击/FPS", "免费游玩/Free to Play"],
        "tags": ["Counter-Strike 2", "竞技/Competitive", "团队/Team-Based", "多人/Multiplayer", "PvP", "射击/Shooter"],
        "price_usd": 0.0,
        "discount_percent": 0,
        "rating": 8.6,
        "release_date": "2023-09-27",
        "platforms": ["windows", "linux"],
        "modes": ["multiplayer"],
        "min_specs": {
            "windows": {"cpu": "4线程", "ram_gb": 8, "gpu": "支持DirectX 11"},
            "linux": {"cpu": "4线程", "ram_gb": 8, "gpu": "支持Vulkan"},
        },
    },
    {
        "app_id": 570,
        "name": "刀塔2",
        "genres": ["MOBA", "免费游玩/Free to Play"],
        "tags": ["Dota 2", "竞技/Competitive", "团队/Team-Based", "多人/Multiplayer", "策略/Strategy"],
        "price_usd": 0.0,
        "discount_percent": 0,
        "rating": 9.1,
        "release_date": "2013-07-09",
        "platforms": ["windows", "mac", "linux"],
        "modes": ["multiplayer"],
        "min_specs": {
            "windows": {"cpu": "双核", "ram_gb": 4, "gpu": "支持DirectX 9"},
            "mac": {"cpu": "双核", "ram_gb": 4, "gpu": "支持Metal"},
            "linux": {"cpu": "双核", "ram_gb": 4, "gpu": "支持OpenGL"},
        },
    },
    {
        "app_id": 578080,
        "name": "绝地求生：大逃杀",
        "genres": ["射击/FPS", "大逃杀/Battle Royale", "免费游玩/Free to Play"],
        "tags": ["PUBG: BATTLEGROUNDS", "生存/Survival", "PvP", "多人/Multiplayer", "战术/Tactical"],
        "price_usd": 0.0,
        "discount_percent": 0,
        "rating": 7.8,
        "release_date": "2017-12-21",
        "platforms": ["windows"],
        "modes": ["multiplayer"],
        "min_specs": {
            "windows": {"cpu": "四核", "ram_gb": 8, "gpu": "GTX 960 / RX 460"},
        },
    },
    {
        "app_id": 252490,
        "name": "腐蚀",
        "genres": ["生存/Survival", "建造/Building", "制作/Crafting"],
        "tags": ["Rust", "开放世界/Open World", "PVP", "多人/Multiplayer", "硬核/Hardcore"],
        "price_usd": 39.99,
        "discount_percent": 40,  # 模拟折扣
        "rating": 8.7,
        "release_date": "2018-02-08",
        "platforms": ["windows"],
        "modes": ["multiplayer"],
        "min_specs": {
            "windows": {"cpu": "四核", "ram_gb": 10, "gpu": "GTX 670 / HD 7870"},
        },
    },
    {
        "app_id": 105600,
        "name": "泰拉瑞亚",
        "genres": ["沙盒/Sandbox", "生存/Survival", "冒险/Adventure"],
        "tags": ["Terraria", "2D", "建造/Building", "探索/Exploration", "多人/Multiplayer", "单人/Singleplayer"],
        "price_usd": 9.99,
        "discount_percent": 50,  # 模拟折扣
        "rating": 9.6,
        "release_date": "2011-05-16",
        "platforms": ["windows", "mac", "linux"],
        "modes": ["singleplayer", "multiplayer"],
        "min_specs": {
            "windows": {"cpu": "双核", "ram_gb": 2, "gpu": "支持DirectX 9"},
            "mac": {"cpu": "双核", "ram_gb": 2, "gpu": "支持OpenGL"},
            "linux": {"cpu": "双核", "ram_gb": 2, "gpu": "支持OpenGL"},
        },
    },
    {
        "app_id": 1172470,
        "name": "Apex英雄",
        "genres": ["射击/FPS", "大逃杀/Battle Royale", "免费游玩/Free to Play"],
        "tags": ["Apex Legends", "英雄射击/Hero Shooter", "团队/Team-Based", "多人/Multiplayer", "PvP"],
        "price_usd": 0.0,
        "discount_percent": 0,
        "rating": 8.1,
        "release_date": "2020-11-04",
        "platforms": ["windows"],
        "modes": ["multiplayer"],
        "min_specs": {
            "windows": {"cpu": "四核", "ram_gb": 8, "gpu": "GTX 970 / R9 290"},
        },
    },
    {
        "app_id": 271590,
        "name": "侠盗猎车手5",
        "genres": ["动作/Action", "开放世界/Open World"],
        "tags": ["Grand Theft Auto V", "犯罪/Crime", "第三人称/Third Person", "单人/Singleplayer", "线上/GTA Online"],
        "price_usd": 29.99,
        "discount_percent": 60,  # 模拟折扣
        "rating": 8.9,
        "release_date": "2015-04-14",
        "platforms": ["windows"],
        "modes": ["singleplayer", "multiplayer"],
        "min_specs": {
            "windows": {"cpu": "四核", "ram_gb": 8, "gpu": "GTX 660 / HD 7870"},
        },
    },
    {
        "app_id": 1086940,
        "name": "博德之门3",
        "genres": ["角色扮演/RPG", "策略/Strategy"],
        "tags": ["Baldur's Gate 3", "剧情丰富/Story Rich", "回合制/Turn-Based", "可联机/Co-op", "单人/Singleplayer"],
        "price_usd": 59.99,
        "discount_percent": 10,  # 模拟折扣
        "rating": 9.6,
        "release_date": "2023-08-03",
        "platforms": ["windows", "mac"],
        "modes": ["singleplayer", "multiplayer"],
        "min_specs": {
            "windows": {"cpu": "四核", "ram_gb": 8, "gpu": "GTX 970 / RX 480"},
            "mac": {"cpu": "Apple Silicon/Intel", "ram_gb": 8, "gpu": "Metal支持"},
        },
    },
    {
        "app_id": 1245620,
        "name": "艾尔登法环",
        "genres": ["角色扮演/RPG", "动作/Action", "开放世界/Open World"],
        "tags": ["ELDEN RING", "魂类/Souls-like", "高难度/Difficult", "单人/Singleplayer", "多人/Multiplayer"],
        "price_usd": 59.99,
        "discount_percent": 20,  # 模拟折扣
        "rating": 9.2,
        "release_date": "2022-02-25",
        "platforms": ["windows"],
        "modes": ["singleplayer", "multiplayer"],
        "min_specs": {
            "windows": {"cpu": "四核", "ram_gb": 12, "gpu": "GTX 1060 / RX 580"},
        },
    },
    {
        "app_id": 1623730,
        "name": "幻兽帕鲁",
        "genres": ["生存/Survival", "制作/Crafting", "开放世界/Open World"],
        "tags": ["Palworld", "多人联机/Online Co-Op", "建造/Building", "收集/Creature Collector", "单人/Singleplayer"],
        "price_usd": 29.99,
        "discount_percent": 0,
        "rating": 9.0,
        "release_date": "2024-01-19",
        "platforms": ["windows"],
        "modes": ["singleplayer", "multiplayer"],
        "min_specs": {
            "windows": {"cpu": "四核", "ram_gb": 16, "gpu": "GTX 1050 / RX 580"},
        },
    },
]


def _final_price(g: dict) -> float:
    p = float(g.get("price_usd", 0.0))
    d = int(g.get("discount_percent", 0))
    if p <= 0 or d <= 0:
        return round(p, 2)
    return round(p * (100 - d) / 100.0, 2)


def _find_game_by_id(app_id: int):
    for g in STEAM_CATALOG:
        if g["app_id"] == app_id:
            return g
    return None


# ==================== 工具函数定义（8个） ====================

def search_games(keyword: str, limit: int = 5):
    """按关键词在游戏名/类型/标签里搜索游戏，返回匹配列表"""
    kw = (keyword or "").strip().lower()
    hits = []
    for g in STEAM_CATALOG:
        hay = " ".join([g["name"], " ".join(g["genres"]), " ".join(g["tags"])]).lower()
        if kw and kw in hay:
            hits.append({
                "app_id": g["app_id"],
                "name": g["name"],
                "price_usd": g["price_usd"],
                "discount_percent": g.get("discount_percent", 0),
                "final_price_usd": _final_price(g),
                "rating": g["rating"],
                "genres": g["genres"][:3],
                "tags": g["tags"][:6],
            })
    hits.sort(key=lambda x: (-x["rating"], x["final_price_usd"]))
    return json.dumps(hits[: max(1, min(limit, 20))], ensure_ascii=False)


def get_game_detail(app_id: int):
    """获取指定游戏的详细信息"""
    g = _find_game_by_id(app_id)
    if not g:
        return json.dumps({"error": "未找到该游戏(app_id不存在)"}, ensure_ascii=False)

    detail = {
        "app_id": g["app_id"],
        "name": g["name"],
        "genres": g["genres"],
        "tags": g["tags"],
        "price_usd": g["price_usd"],
        "discount_percent": g.get("discount_percent", 0),
        "final_price_usd": _final_price(g),
        "rating": g["rating"],
        "release_date": g["release_date"],
        "platforms": g["platforms"],
        "modes": g["modes"],
        "min_specs": g["min_specs"],
        "note": "此为本地模拟数据；接真实Steam API时可替换该工具实现",
    }
    return json.dumps(detail, ensure_ascii=False)


def list_top_games(sort_by: str = "rating", limit: int = 10, genre: str = None, max_price_usd: float = None):
    """
    返回“高分/低价/免费/某类型”的游戏榜单
    sort_by: rating | price_low | price_high | free
    """
    games = STEAM_CATALOG[:]

    if genre:
        ge = genre.strip().lower()
        games = [g for g in games if any(ge in x.lower() for x in g["genres"])]

    if max_price_usd is not None:
        games = [g for g in games if _final_price(g) <= float(max_price_usd)]

    sb = (sort_by or "rating").strip().lower()
    if sb == "free":
        games = [g for g in games if _final_price(g) == 0.0]
        games.sort(key=lambda g: -g["rating"])
    elif sb == "price_low":
        games.sort(key=lambda g: (_final_price(g), -g["rating"]))
    elif sb == "price_high":
        games.sort(key=lambda g: (-_final_price(g), -g["rating"]))
    else:  # rating
        games.sort(key=lambda g: -g["rating"])

    out = [{
        "app_id": g["app_id"],
        "name": g["name"],
        "price_usd": g["price_usd"],
        "discount_percent": g.get("discount_percent", 0),
        "final_price_usd": _final_price(g),
        "rating": g["rating"],
        "genres": g["genres"][:3],
        "tags": g["tags"][:6],
    } for g in games[: max(1, min(limit, 20))]]

    return json.dumps(out, ensure_ascii=False)


def recommend_games(preferred_genres: list = None,
                    preferred_tags: list = None,
                    platform: str = "windows",
                    multiplayer: bool = None,
                    max_price_usd: float = 60.0,
                    limit: int = 5):
    """按偏好推荐游戏：类型/标签/平台/是否多人/预算"""
    preferred_genres = preferred_genres or []
    preferred_tags = preferred_tags or []
    platform = (platform or "windows").strip().lower()

    def score(g):
        s = 0.0
        for pg in preferred_genres:
            if any(pg.lower() in x.lower() for x in g["genres"]):
                s += 2.0
        for pt in preferred_tags:
            if any(pt.lower() in x.lower() for x in g["tags"]):
                s += 1.5
        s += g["rating"] / 10.0
        if _final_price(g) == 0:
            s += 0.5
        if int(g.get("discount_percent", 0)) >= 40:
            s += 0.2
        return s

    candidates = []
    for g in STEAM_CATALOG:
        if platform not in g["platforms"]:
            continue
        if _final_price(g) > float(max_price_usd):
            continue
        if multiplayer is True and "multiplayer" not in g["modes"]:
            continue
        if multiplayer is False and "singleplayer" not in g["modes"]:
            continue
        candidates.append(g)

    candidates.sort(key=lambda g: (-score(g), _final_price(g)))
    out = [{
        "app_id": g["app_id"],
        "name": g["name"],
        "price_usd": g["price_usd"],
        "discount_percent": g.get("discount_percent", 0),
        "final_price_usd": _final_price(g),
        "rating": g["rating"],
        "genres": g["genres"][:3],
        "tags": g["tags"][:8],
        "platforms": g["platforms"],
        "modes": g["modes"],
        "reason_hint": "按类型/标签/平台/预算/是否多人综合排序（含评分与折扣轻微加权）",
    } for g in candidates[: max(1, min(limit, 20))]]

    if not out:
        return json.dumps({
            "warning": "没有找到满足约束的推荐结果",
            "tips": "可以放宽预算、切换平台、减少偏好标签或取消多人/单人限制",
        }, ensure_ascii=False)

    return json.dumps(out, ensure_ascii=False)


def compare_games(app_ids: list, region: str = "US"):
    """对比多款游戏：价格/评分/类型/平台/模式"""
    rows = []
    for app_id in app_ids:
        try:
            aid = int(app_id) if isinstance(app_id, str) and app_id.strip().isdigit() else app_id
        except Exception:
            aid = app_id

        g = _find_game_by_id(aid)
        if not g:
            rows.append({"app_id": app_id, "error": "未找到该游戏"})
            continue

        rows.append({
            "app_id": g["app_id"],
            "name": g["name"],
            "price_usd": g["price_usd"],
            "discount_percent": g.get("discount_percent", 0),
            "final_price_usd": _final_price(g),
            "rating": g["rating"],
            "genres": g["genres"][:3],
            "platforms": g["platforms"],
            "modes": g["modes"],
        })

    rows.sort(key=lambda x: (x.get("final_price_usd", 10**9), -x.get("rating", 0)))
    return json.dumps({"region": region, "games": rows}, ensure_ascii=False)


# -------- 工具 6：折扣榜单 --------
def list_discounted_games(min_discount_percent: int = 30, max_final_price_usd: float = None, limit: int = 10):
    """列出当前打折的游戏（按折扣优先，其次按最终价低与评分高）"""
    md = int(min_discount_percent) if min_discount_percent is not None else 0
    games = []
    for g in STEAM_CATALOG:
        d = int(g.get("discount_percent", 0))
        if d < md:
            continue
        fp = _final_price(g)
        if max_final_price_usd is not None and fp > float(max_final_price_usd):
            continue
        games.append(g)

    games.sort(key=lambda g: (-int(g.get("discount_percent", 0)), _final_price(g), -g["rating"]))
    out = [{
        "app_id": g["app_id"],
        "name": g["name"],
        "price_usd": g["price_usd"],
        "discount_percent": g.get("discount_percent", 0),
        "final_price_usd": _final_price(g),
        "rating": g["rating"],
        "genres": g["genres"][:3],
    } for g in games[: max(1, min(limit, 20))]]

    return json.dumps(out, ensure_ascii=False)


# -------- 工具 7：相似游戏 --------
def get_similar_games(app_id: int, limit: int = 5):
    """基于类型/标签重叠度找相似游戏"""
    base = _find_game_by_id(app_id)
    if not base:
        return json.dumps({"error": "未找到该游戏(app_id不存在)"}, ensure_ascii=False)

    base_genres = set([x.lower() for x in base["genres"]])
    base_tags = set([x.lower() for x in base["tags"]])

    scored = []
    for g in STEAM_CATALOG:
        if g["app_id"] == base["app_id"]:
            continue
        genres = set([x.lower() for x in g["genres"]])
        tags = set([x.lower() for x in g["tags"]])

        genre_overlap = len(base_genres & genres)
        tag_overlap = len(base_tags & tags)

        s = genre_overlap * 2.0 + tag_overlap * 1.0 + g["rating"] / 10.0
        scored.append((s, genre_overlap, tag_overlap, g))

    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], _final_price(x[3])))
    out = [{
        "app_id": g["app_id"],
        "name": g["name"],
        "final_price_usd": _final_price(g),
        "discount_percent": g.get("discount_percent", 0),
        "rating": g["rating"],
        "genres": g["genres"][:3],
        "tags": g["tags"][:8],
        "why_similar": {
            "genre_overlap_count": go,
            "tag_overlap_count": to,
        }
    } for (s, go, to, g) in scored[: max(1, min(limit, 20))]]

    return json.dumps({
        "base_game": {"app_id": base["app_id"], "name": base["name"]},
        "similar_games": out
    }, ensure_ascii=False)


# -------- 工具 8：电脑配置可玩性粗判 --------
def check_pc_requirements(app_id: int, platform: str = "windows", ram_gb: int = 8, cpu: str = "", gpu: str = ""):
    """
    粗略判断：你的配置是否满足最低要求（仅做演示，不做真实硬件对比）
    - 核心判断：RAM 是否 >= 最低RAM
    - CPU/GPU：给出“文本提示”，不做严格比对
    """
    g = _find_game_by_id(app_id)
    if not g:
        return json.dumps({"error": "未找到该游戏(app_id不存在)"}, ensure_ascii=False)

    pf = (platform or "windows").strip().lower()
    min_specs = g.get("min_specs", {}).get(pf)

    if not min_specs:
        return json.dumps({
            "app_id": app_id,
            "name": g["name"],
            "platform": pf,
            "ok": False,
            "reason": f"该游戏模拟数据中没有 {pf} 的最低配置条目",
        }, ensure_ascii=False)

    required_ram = int(min_specs.get("ram_gb", 0))
    user_ram = int(ram_gb) if ram_gb is not None else 0

    ok_ram = user_ram >= required_ram
    ok = bool(ok_ram)

    guidance = []
    if not ok_ram:
        guidance.append(f"内存不足：你是 {user_ram}GB，最低要求 {required_ram}GB。")
    else:
        guidance.append(f"内存满足：你是 {user_ram}GB，最低要求 {required_ram}GB。")

    if cpu:
        guidance.append(f"CPU你填的是“{cpu}”，最低要求描述为“{min_specs.get('cpu', '')}”（本工具不做严格对比）。")
    else:
        guidance.append(f"最低CPU要求描述为“{min_specs.get('cpu', '')}”。")

    if gpu:
        guidance.append(f"GPU你填的是“{gpu}”，最低要求描述为“{min_specs.get('gpu', '')}”（本工具不做严格对比）。")
    else:
        guidance.append(f"最低GPU要求描述为“{min_specs.get('gpu', '')}”。")

    return json.dumps({
        "app_id": g["app_id"],
        "name": g["name"],
        "platform": pf,
        "ok": ok,
        "min_specs": min_specs,
        "your_input": {"ram_gb": user_ram, "cpu": cpu, "gpu": gpu},
        "guidance": guidance,
        "note": "该判断为演示级粗判；真实可玩性还与显卡型号/驱动/分辨率/画质有关",
    }, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义（8个） ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_games",
            "description": "按关键词搜索Steam游戏（在游戏名/类型/标签中匹配），返回匹配结果列表（含app_id、价格、评分等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词，例如：RPG、开放世界、刀塔、射击、Terraria"},
                    "limit": {"type": "integer", "description": "返回条数，建议1-10"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_detail",
            "description": "获取指定Steam游戏的详细信息（类型、标签、价格、折扣、评分、平台、模式、最低配置等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "integer", "description": "Steam应用ID，例如：570、730、1086940、1245620"}
                },
                "required": ["app_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_top_games",
            "description": "获取榜单：按评分/价格/免费等排序，可指定类型与预算（按最终价final_price计算）",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {"type": "string", "description": "排序方式：rating | price_low | price_high | free"},
                    "limit": {"type": "integer", "description": "返回条数，建议1-20"},
                    "genre": {"type": "string", "description": "筛选类型（可选），例如：RPG、FPS、Open World"},
                    "max_price_usd": {"type": "number", "description": "最高最终价格（可选），例如：30、60"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_games",
            "description": "按用户偏好推荐游戏：类型/标签/平台/是否多人/预算，并返回推荐理由提示",
            "parameters": {
                "type": "object",
                "properties": {
                    "preferred_genres": {"type": "array", "items": {"type": "string"}, "description": "偏好类型列表，例如：[\"RPG\",\"Open World\"]"},
                    "preferred_tags": {"type": "array", "items": {"type": "string"}, "description": "偏好标签列表，例如：[\"剧情丰富\",\"Co-op\"]"},
                    "platform": {"type": "string", "description": "平台：windows/mac/linux，默认windows"},
                    "multiplayer": {"type": "boolean", "description": "是否偏好多人与联机：true/false（可选）"},
                    "max_price_usd": {"type": "number", "description": "预算上限（最终价，美元），默认60"},
                    "limit": {"type": "integer", "description": "返回条数，建议1-10"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_games",
            "description": "对比多款游戏：最终价、评分、类型、平台、模式等，便于做购买/游玩决策",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_ids": {"type": "array", "items": {"type": "integer"}, "description": "要对比的app_id列表，例如：[730, 570, 1086940]"},
                    "region": {"type": "string", "description": "地区（占位字段，默认US）"}
                },
                "required": ["app_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_discounted_games",
            "description": "列出当前打折的热门游戏（按折扣优先），可限制最低折扣与最高最终价",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_discount_percent": {"type": "integer", "description": "最低折扣百分比，例如：30 表示至少打7折"},
                    "max_final_price_usd": {"type": "number", "description": "最高最终价（可选），例如：20、30"},
                    "limit": {"type": "integer", "description": "返回条数，建议1-20"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_games",
            "description": "找与某游戏相似的游戏（按类型/标签重叠度 + 评分综合排序）",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "integer", "description": "基准游戏的app_id，例如：1245620（艾尔登法环）"},
                    "limit": {"type": "integer", "description": "返回条数，建议1-10"}
                },
                "required": ["app_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_pc_requirements",
            "description": "粗略判断你的电脑配置是否达到某游戏最低要求（演示级，不做真实硬件对比）",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "integer", "description": "游戏app_id"},
                    "platform": {"type": "string", "description": "windows/mac/linux，默认windows"},
                    "ram_gb": {"type": "integer", "description": "你的内存大小（GB），例如：8、16"},
                    "cpu": {"type": "string", "description": "你的CPU描述（可选），例如：i5-8400"},
                    "gpu": {"type": "string", "description": "你的GPU描述（可选），例如：GTX 1060"}
                },
                "required": ["app_id"]
            }
        }
    }
]


# ==================== 工具注册 ====================

available_functions = {
    "search_games": search_games,
    "get_game_detail": get_game_detail,
    "list_top_games": list_top_games,
    "recommend_games": recommend_games,
    "compare_games": compare_games,
    "list_discounted_games": list_discounted_games,
    "get_similar_games": get_similar_games,
    "check_pc_requirements": check_pc_requirements,
}


def _coerce_args(function_name: str, args: dict) -> dict:
    """把模型生成的参数做最小强制转换，防止类型错导致工具报错。"""

    # int
    for k in ("limit", "app_id", "min_discount_percent", "ram_gb"):
        if k in args and isinstance(args[k], str):
            s = args[k].strip()
            if s.isdigit():
                args[k] = int(s)

    # float
    for k in ("max_price_usd", "max_final_price_usd"):
        if k in args and isinstance(args[k], str):
            s = args[k].strip()
            try:
                args[k] = float(s)
            except Exception:
                pass

    # compare_games：有时模型把列表生成成逗号分隔字符串
    if function_name == "compare_games" and "app_ids" in args:
        if isinstance(args["app_ids"], str):
            parts = [x.strip() for x in args["app_ids"].split(",") if x.strip()]
            coerced = []
            for p in parts:
                if p.isdigit():
                    coerced.append(int(p))
                else:
                    coerced.append(p)
            args["app_ids"] = coerced

    # recommend_games：有时把 array 写成单个字符串
    if function_name == "recommend_games":
        for k in ("preferred_genres", "preferred_tags"):
            if k in args and isinstance(args[k], str):
                args[k] = [x.strip() for x in args[k].split(",") if x.strip()]

    return args


# ==================== Agent核心逻辑 ====================

def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("未检测到API Key：请设置 DASHSCOPE_API_KEY / OPENAI_API_KEY 或在 run_agent 传 api_key")

    client = OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {
            "role": "system",
            "content": """你是一位Steam游戏平台助手。你可以：
1. 帮用户找游戏（按关键词、类型、标签）
2. 给出游戏详情（价格/折扣/评分/平台/模式/最低配置）
3. 输出榜单（高分、免费、低价等）
4. 按用户偏好做推荐（类型/标签/平台/预算/是否多人）
5. 对比多款游戏，帮助用户做购买/游玩决策
6. 列出当前折扣游戏（打折力度/最终价）
7. 找相似游戏（类型/标签重叠度）
8. 粗判电脑配置是否够用（演示级）

请根据用户问题，优先使用合适的工具获取信息，然后用清晰的结构化中文给出建议。
如果用户信息不足（例如平台/预算/偏好），可以先追问1-2个关键问题再推荐。"""
        },
        {"role": "user", "content": user_query}
    ]

    print("\n" + "=" * 60)
    print("【用户问题】")
    print(user_query)
    print("=" * 60)

    max_iterations = 5
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message.model_dump(exclude_none=True))

        tool_calls = getattr(response_message, "tool_calls", None) or []
        if not tool_calls:
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            raw_args = tool_call.function.arguments

            try:
                function_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                function_args = {}

            function_args = _coerce_args(function_name, function_args)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            if function_name not in available_functions:
                err = json.dumps({"error": f"未找到工具 {function_name}"}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": err})
                continue

            try:
                function_response = available_functions[function_name](**function_args)
            except TypeError as e:
                function_response = json.dumps({"error": f"工具参数错误: {str(e)}"}, ensure_ascii=False)
            except Exception as e:
                function_response = json.dumps({"error": f"工具执行异常: {str(e)}"}, ensure_ascii=False)

            print(f"工具返回: {function_response[:200]}..." if len(function_response) > 200 else f"工具返回: {function_response}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_response
            })

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"


# ==================== 示例场景 ====================

def demo_scenarios():
    print("\n" + "#" * 60)
    print("# Steam游戏平台Agent演示 - Function Call能力展示（8 tools）")
    print("#" * 60)

    scenarios = [
        # search_games
        "帮我搜一下“开放世界RPG”，给我几款评分高的",
        # recommend_games
        "我想要剧情丰富的RPG，预算60美元以内，Windows，最好能联机",
        # get_game_detail
        "博德之门3（1086940）多少钱？支持哪些平台？最低配置是什么？",
        # list_top_games
        "列一个免费游戏高分榜 Top 5",
        # compare_games
        "对比一下 1245620、1086940、271590：最终价、评分、平台、是否多人",
        # list_discounted_games
        "最近有哪些折扣力度>=40%的热门游戏？给我Top 5",
        # get_similar_games
        "找几款和“艾尔登法环”(1245620) 类似的游戏，说明相似点",
        # check_pc_requirements
        "我电脑是Windows，内存8GB，显卡GTX 1060，能玩艾尔登法环(1245620)吗？"
    ]

    print("\n以下是几个示例问题（已覆盖8个工具），你可以直接复制运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")

    print("\n" + "-" * 60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-" * 60)


if __name__ == "__main__":
    # demo_scenarios()

    # 运行示例（取消注释下面的代码来运行）
    # 示例1：折扣榜单（新增工具）
    run_agent("最近有哪些折扣力度>=40%的热门游戏？给我Top 5", model="qwen-plus")

    # 示例2：找相似游戏（新增工具）
    # run_agent("找几款和“艾尔登法环”(1245620) 类似的游戏，说明相似点", model="qwen-plus")

    # 示例3：配置可玩性（新增工具）
    # run_agent("我电脑是Windows，内存8GB，显卡GTX 1060，能玩艾尔登法环(1245620)吗？", model="qwen-plus")

    # 示例4：搜索
    # run_agent("帮我搜一下“开放世界RPG”，给我几款评分高的", model="qwen-plus")

    # 示例5：推荐
    # run_agent("我想要剧情丰富的RPG，预算60美元以内，Windows，最好能联机", model="qwen-plus")

    # 示例6：详情
    # run_agent("博德之门3（1086940）多少钱？支持哪些平台？最低配置是什么？", model="qwen-plus")

    # 示例7：榜单
    # run_agent("列一个免费游戏高分榜 Top 5", model="qwen-plus")

    # 示例8：对比
    # run_agent("对比一下 1245620、1086940、271590：最终价、评分、平台、是否多人", model="qwen-plus")
