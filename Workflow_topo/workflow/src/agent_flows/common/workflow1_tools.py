"""模拟工具：搜索酒店和支付功能"""

import random
from typing import Dict, List, Tuple


# 模拟数据库：不同街道的酒店信息
HOTEL_DATABASE = {
    "和平路": [
        {"name": "和平大酒店", "price": 280, "rating": 4.5, "address": "和平路123号"},
        {"name": "和平商务酒店", "price": 220, "rating": 4.2, "address": "和平路456号"},
        {"name": "和平快捷酒店", "price": 150, "rating": 3.8, "address": "和平路789号"},
        {"name": "和平豪华酒店", "price": 450, "rating": 4.8, "address": "和平路321号"},
    ],
    "中山大道": [
        {"name": "中山国际酒店", "price": 350, "rating": 4.6, "address": "中山大道100号"},
        {"name": "中山商旅酒店", "price": 260, "rating": 4.3, "address": "中山大道200号"},
        {"name": "中山快捷连锁", "price": 180, "rating": 3.9, "address": "中山大道300号"},
    ],
    "解放路": [
        {"name": "解放大酒店", "price": 290, "rating": 4.4, "address": "解放路50号"},
        {"name": "解放商务中心", "price": 240, "rating": 4.1, "address": "解放路80号"},
    ],
    "人民路": [
        {"name": "人民宾馆", "price": 200, "rating": 4.0, "address": "人民路10号"},
        {"name": "人民大酒店", "price": 320, "rating": 4.5, "address": "人民路20号"},
        {"name": "人民快捷酒店", "price": 160, "rating": 3.7, "address": "人民路30号"},
    ],
}


def search_hotels(street: str) -> List[Dict[str, any]]:
    """
    模拟搜索酒店工具

    Args:
        street: 街道名称

    Returns:
        酒店列表，如果找不到返回空列表
    """
    # 模拟网络延迟和计算
    street = street.strip()

    # 模糊匹配：如果用户输入的街道包含在数据库中的某个街道名称中
    results = []
    for db_street, hotels in HOTEL_DATABASE.items():
        if street in db_street or db_street in street:
            results.extend(hotels)
            break

    # 如果没有精确匹配，尝试部分匹配
    if not results:
        for db_street, hotels in HOTEL_DATABASE.items():
            # 检查是否有共同字符
            common_chars = set(street) & set(db_street)
            if len(common_chars) >= 2:  # 至少有2个共同字
                results.extend(hotels)
                break

    return results


def book_and_pay(
    hotel: Dict[str, any],
    secure_environment: bool = False
) -> Tuple[bool, str]:
    """
    模拟预订和支付工具

    这个工具需要在特殊的安全执行环境中运行，只有支付节点才有这个环境

    Args:
        hotel: 酒店信息字典
        secure_environment: 是否在安全环境中执行

    Returns:
        (成功标志, 消息)
    """
    # 检查是否在安全环境中
    if not secure_environment:
        return False, "支付失败：未在安全执行环境中，无法完成支付操作"

    # 检查酒店信息是否完整
    if not hotel or not isinstance(hotel, dict):
        return False, "支付失败：酒店信息无效"

    hotel_name = hotel.get("name", "未知酒店")
    price = hotel.get("price", 0)

    if price <= 0:
        return False, f"支付失败：{hotel_name}的价格信息无效"

    # 模拟支付过程，有10%的概率失败（模拟网络问题、余额不足等情况）
    success_rate = 0.9
    if random.random() < success_rate:
        order_id = f"ORD{random.randint(100000, 999999)}"
        return True, f"支付成功！已预订{hotel_name}，房间价格¥{price}，订单号：{order_id}"
    else:
        # 模拟不同的失败原因
        failure_reasons = [
            "网络连接超时，请稍后重试",
            "支付接口响应异常，请检查账户余额",
            "银行系统维护中，请使用其他支付方式",
            "账户余额不足，请充值后重试",
        ]
        reason = random.choice(failure_reasons)
        return False, f"支付失败：{reason}"


def validate_hotel_availability(hotel: Dict[str, any], nights: int = 1) -> Tuple[bool, str]:
    """
    模拟检查酒店可用性的工具（可以作为额外的验证步骤）

    Args:
        hotel: 酒店信息
        nights: 入住天数

    Returns:
        (可用标志, 消息)
    """
    if not hotel:
        return False, "酒店信息无效"

    hotel_name = hotel.get("name", "未知酒店")

    # 90%的概率有房
    if random.random() < 0.9:
        return True, f"{hotel_name}有空房，可预订{nights}晚"
    else:
        return False, f"{hotel_name}暂无空房，请选择其他酒店"
