"""LangGraph工作流：智能管家助手 - 酒店预订系统

这个工作流包含7个节点，模拟一个智能管家帮助用户搜索和预订酒店的完整流程。
每个节点代表一个智能体，具有不同的职责和资源使用特点。
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..common.tools import book_and_pay, search_hotels, validate_hotel_availability
from ..common.utils import llm


# ============ 状态定义 ============
class BookingState(TypedDict, total=False):
    """工作流状态，记录整个预订流程的数据"""
    user_request: str                           # 用户原始请求
    understood_intent: str                      # 理解的用户意图
    parsed_params: Dict[str, Any]              # 解析出的参数
    search_strategy: str                        # 搜索策略
    search_results: List[Dict[str, Any]]       # 搜索结果
    filtered_results: List[Dict[str, Any]]     # 筛选后的结果
    selected_hotel: Optional[Dict[str, Any]]   # 选中的酒店
    availability_check: bool                    # 可用性检查结果
    payment_status: Literal["pending", "success", "failed"]  # 支付状态
    payment_message: str                        # 支付消息
    error_message: str                          # 错误消息
    reasoning_log: List[str]                    # 推理日志
    final_response: str                         # 最终回复


# ============ 辅助函数 ============
def _append_log(state: BookingState, note: str) -> BookingState:
    """向状态中添加日志"""
    log = state.get("reasoning_log", []) + [note]
    state["reasoning_log"] = log
    return state


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """从文本中提取JSON对象"""
    if isinstance(text, dict):
        return text

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从代码块中提取
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个完整的JSON对象
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _fallback_parse(user_request: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """当LLM解析失败时，使用启发式规则解析用户请求"""
    parsed = dict(parsed)

    # 检测街道名称
    common_streets = ["和平路", "中山大道", "解放路", "人民路", "建设路", "友谊路"]
    for street in common_streets:
        if street in user_request and not parsed.get("street"):
            parsed["street"] = street
            break

    # 提取价格范围（如：200-300、200到300、200至300）
    budget_patterns = [
        r'(\d{2,5})\s*[-到至~]\s*(\d{2,5})',
        r'(\d{2,5})\s*元.*?(\d{2,5})\s*元',
    ]
    for pattern in budget_patterns:
        match = re.search(pattern, user_request)
        if match:
            low, high = match.groups()
            parsed.setdefault("budget_min", int(low))
            parsed.setdefault("budget_max", int(high))
            break

    # 提取入住天数
    nights_match = re.search(r'(\d+)\s*[晚天夜]', user_request)
    if nights_match:
        parsed.setdefault("nights", int(nights_match.group(1)))
    else:
        parsed.setdefault("nights", 1)

    # 提取房间数量
    rooms_match = re.search(r'(\d+)\s*[间个]房', user_request)
    if rooms_match:
        parsed.setdefault("rooms", int(rooms_match.group(1)))
    else:
        parsed.setdefault("rooms", 1)

    return parsed


# ============ 节点1：意图理解节点 ============
def intent_understanding_node(state: BookingState) -> BookingState:
    """
    节点1：意图理解智能体
    职责：理解用户的整体意图，判断用户想做什么
    资源特点：主要使用LLM推理，中等CPU使用
    """
    user_request = state.get("user_request", "")

    # 构建提示词
    prompt = [
        SystemMessage(content="""你是意图理解智能体。
你的任务是理解用户的整体意图，用一句话总结用户想要做什么。

例如：
- 用户输入："帮我找和平路的酒店" → 意图："用户想在和平路搜索酒店"
- 用户输入："订一个200块左右的房间" → 意图："用户想预订价格在200元左右的酒店房间"

请用简洁的中文回复用户的意图。"""),
        HumanMessage(content=user_request)
    ]

    try:
        ai_msg = llm.invoke(prompt)
        intent = getattr(ai_msg, "content", "未能理解用户意图")
        state["understood_intent"] = intent
        _append_log(state, f"[节点1-意图理解] {intent}")
    except Exception as e:
        state["understood_intent"] = "意图理解失败"
        _append_log(state, f"[节点1-意图理解] 错误: {str(e)}")

    return state


# ============ 节点2：参数提取节点 ============
def parameter_extraction_node(state: BookingState) -> BookingState:
    """
    节点2：参数提取智能体
    职责：从用户请求中提取结构化参数（街道、价格、天数等）
    资源特点：使用LLM+正则表达式，中等CPU和内存使用
    """
    user_request = state.get("user_request", "")

    # 构建提示词
    prompt = [
        SystemMessage(content="""你是参数提取智能体。
从用户的请求中提取以下参数，并以JSON格式返回：
- street: 街道名称（字符串）
- budget_min: 最低预算（整数，单位：元）
- budget_max: 最高预算（整数，单位：元）
- nights: 入住天数（整数，默认1）
- rooms: 房间数量（整数，默认1）

如果某个参数无法确定，设为null。

示例输出：
```json
{
    "street": "和平路",
    "budget_min": 200,
    "budget_max": 300,
    "nights": 1,
    "rooms": 1
}
```
"""),
        HumanMessage(content=user_request)
    ]

    try:
        ai_msg = llm.invoke(prompt)
        content = getattr(ai_msg, "content", "{}")
        parsed = _parse_json_from_text(content)

        # 使用启发式规则作为后备
        parsed = _fallback_parse(user_request, parsed)

        state["parsed_params"] = parsed
        _append_log(state, f"[节点2-参数提取] 提取参数: {parsed}")
    except Exception as e:
        state["parsed_params"] = {}
        _append_log(state, f"[节点2-参数提取] 错误: {str(e)}")

    return state


# ============ 节点3：搜索策略规划节点 ============
def search_planning_node(state: BookingState) -> BookingState:
    """
    节点3：搜索策略规划智能体
    职责：根据提取的参数，规划如何搜索酒店
    资源特点：轻量级LLM调用，低资源使用
    """
    parsed_params = state.get("parsed_params", {})
    street = parsed_params.get("street", "未指定")
    budget = f"{parsed_params.get('budget_min', '不限')}-{parsed_params.get('budget_max', '不限')}元"

    prompt = [
        SystemMessage(content="""你是搜索策略规划智能体。
根据用户需求，简要说明搜索策略（不超过50字）。

例如：
- "将在和平路搜索酒店，重点关注200-300元价位的选项"
- "在中山大道范围内全面搜索，不限价格"
"""),
        HumanMessage(content=f"街道：{street}，预算：{budget}")
    ]

    try:
        ai_msg = llm.invoke(prompt)
        strategy = getattr(ai_msg, "content", "标准搜索策略")
        state["search_strategy"] = strategy
        _append_log(state, f"[节点3-搜索规划] {strategy}")
    except Exception as e:
        state["search_strategy"] = "默认搜索策略"
        _append_log(state, f"[节点3-搜索规划] 错误: {str(e)}")

    return state


# ============ 节点4：酒店搜索节点 ============
def hotel_search_node(state: BookingState) -> BookingState:
    """
    节点4：酒店搜索智能体
    职责：调用搜索工具，获取酒店列表
    资源特点：涉及外部工具调用（模拟），中等I/O和内存使用
    """
    parsed_params = state.get("parsed_params", {})
    street = parsed_params.get("street", "")

    if not street:
        state["search_results"] = []
        state["error_message"] = "未指定搜索街道"
        _append_log(state, "[节点4-酒店搜索] 错误：未指定街道")
        return state

    try:
        # 调用搜索工具
        results = search_hotels(street)
        state["search_results"] = results

        if results:
            _append_log(state, f"[节点4-酒店搜索] 在{street}找到{len(results)}家酒店")
        else:
            _append_log(state, f"[节点4-酒店搜索] 在{street}未找到酒店")
            state["error_message"] = f"很抱歉，在{street}没有找到酒店信息"
    except Exception as e:
        state["search_results"] = []
        state["error_message"] = f"搜索过程出错：{str(e)}"
        _append_log(state, f"[节点4-酒店搜索] 错误: {str(e)}")

    return state


# ============ 节点5：结果筛选与选择节点 ============
def filter_and_selection_node(state: BookingState) -> BookingState:
    """
    节点5：结果筛选与选择智能体
    职责：根据预算筛选酒店，并选择最合适的一家
    资源特点：数据处理+LLM推理，较高CPU和内存使用
    """
    search_results = state.get("search_results", [])
    parsed_params = state.get("parsed_params", {})

    if not search_results:
        state["filtered_results"] = []
        state["selected_hotel"] = None
        _append_log(state, "[节点5-筛选选择] 无搜索结果可筛选")
        return state

    # 步骤1：根据预算筛选
    budget_min = parsed_params.get("budget_min")
    budget_max = parsed_params.get("budget_max")

    if budget_min is not None and budget_max is not None:
        filtered = [
            hotel for hotel in search_results
            if budget_min <= hotel.get("price", 0) <= budget_max
        ]
        state["filtered_results"] = filtered
        _append_log(state, f"[节点5-筛选选择] 按预算{budget_min}-{budget_max}元筛选，剩余{len(filtered)}家")
    else:
        filtered = search_results
        state["filtered_results"] = filtered
        _append_log(state, "[节点5-筛选选择] 未设置预算，保留所有结果")

    if not filtered:
        state["selected_hotel"] = None
        _append_log(state, "[节点5-筛选选择] 筛选后无符合条件的酒店")
        return state

    # 步骤2：使用LLM选择最合适的酒店
    prompt = [
        SystemMessage(content="""你是酒店选择智能体。
从候选酒店列表中选择最合适的一家，综合考虑价格、评分和位置。
返回JSON格式，包含：
- name: 酒店名称
- reason: 选择理由（简短）

示例：
```json
{
    "name": "和平商务酒店",
    "reason": "价格适中且评分较高"
}
```
"""),
        HumanMessage(content=f"候选酒店：\n{json.dumps(filtered, ensure_ascii=False, indent=2)}")
    ]

    try:
        ai_msg = llm.invoke(prompt)
        content = getattr(ai_msg, "content", "{}")
        choice = _parse_json_from_text(content)
        chosen_name = choice.get("name")
        reason = choice.get("reason", "综合评估最优")

        # 在筛选结果中查找选中的酒店
        selected = None
        for hotel in filtered:
            if hotel.get("name") == chosen_name:
                selected = hotel
                break

        # 如果没找到，默认选第一个
        if not selected and filtered:
            selected = filtered[0]
            reason = "默认选择评分最高的酒店"

        state["selected_hotel"] = selected
        if selected:
            _append_log(state, f"[节点5-筛选选择] 选定：{selected.get('name')} - {reason}")
        else:
            _append_log(state, "[节点5-筛选选择] 未能选定酒店")
    except Exception as e:
        # 出错时选择第一个
        if filtered:
            state["selected_hotel"] = filtered[0]
            _append_log(state, f"[节点5-筛选选择] 选择过程出错，默认选择：{filtered[0].get('name')}")
        else:
            state["selected_hotel"] = None
            _append_log(state, f"[节点5-筛选选择] 错误: {str(e)}")

    return state


# ============ 节点6：支付节点（特殊安全环境）============
def payment_node(state: BookingState) -> BookingState:
    """
    节点6：支付智能体
    职责：在特殊安全环境中执行支付操作
    资源特点：需要特殊执行环境，模拟安全支付流程，涉及加密操作等
    特殊性：只有这个节点具有支付权限和安全环境
    """
    selected_hotel = state.get("selected_hotel")

    if not selected_hotel:
        state["payment_status"] = "failed"
        state["payment_message"] = "没有选定的酒店，无法执行支付"
        _append_log(state, "[节点6-支付] 支付失败：无选定酒店")
        return state

    # 先验证酒店可用性
    parsed_params = state.get("parsed_params", {})
    nights = parsed_params.get("nights", 1)

    try:
        available, msg = validate_hotel_availability(selected_hotel, nights)
        state["availability_check"] = available

        if not available:
            state["payment_status"] = "failed"
            state["payment_message"] = f"预订失败：{msg}"
            _append_log(state, f"[节点6-支付] {msg}")
            return state

        _append_log(state, f"[节点6-支付] 可用性检查通过：{msg}")
    except Exception as e:
        _append_log(state, f"[节点6-支付] 可用性检查出错: {str(e)}")
        # 继续尝试支付

    # 执行支付（在特殊安全环境中）
    try:
        # 模拟：只有此节点拥有安全执行环境
        secure_environment = True
        success, payment_msg = book_and_pay(selected_hotel, secure_environment=secure_environment)

        state["payment_status"] = "success" if success else "failed"
        state["payment_message"] = payment_msg

        if success:
            _append_log(state, f"[节点6-支付] {payment_msg}")
        else:
            _append_log(state, f"[节点6-支付] 支付失败：{payment_msg}")
    except Exception as e:
        state["payment_status"] = "failed"
        state["payment_message"] = f"支付过程异常：{str(e)}"
        _append_log(state, f"[节点6-支付] 异常: {str(e)}")

    return state


# ============ 节点7：结果总结节点 ============
def summarization_node(state: BookingState) -> BookingState:
    """
    节点7：总结智能体
    职责：汇总整个流程的结果，生成用户友好的回复
    资源特点：LLM文本生成，较高CPU使用
    """
    # 收集所有关键信息
    user_request = state.get("user_request", "")
    understood_intent = state.get("understood_intent", "")
    parsed_params = state.get("parsed_params", {})
    search_results = state.get("search_results", [])
    filtered_results = state.get("filtered_results", [])
    selected_hotel = state.get("selected_hotel")
    payment_status = state.get("payment_status", "pending")
    payment_message = state.get("payment_message", "")
    error_message = state.get("error_message", "")
    reasoning_log = state.get("reasoning_log", [])

    # 构建总结提示
    summary_data = {
        "用户请求": user_request,
        "理解意图": understood_intent,
        "提取参数": parsed_params,
        "搜索到酒店数": len(search_results),
        "筛选后酒店数": len(filtered_results),
        "选定酒店": selected_hotel.get("name") if selected_hotel else "无",
        "酒店详情": selected_hotel if selected_hotel else None,
        "支付状态": payment_status,
        "支付消息": payment_message,
        "错误消息": error_message,
    }

    prompt = [
        SystemMessage(content="""你是总结智能体。
根据工作流的执行结果，生成一个用户友好的中文回复。

要求：
1. 说明搜索了什么、找到了多少酒店
2. 说明筛选和选择的结果
3. 说明支付结果
4. 如果有错误或失败，友好地解释原因，并给出建议
5. 语气要专业、友好、有礼貌
6. 回复要简洁明了，不超过200字

示例（成功）：
"根据您的要求，我在和平路搜索了酒店，找到4家候选。按照200-300元的预算筛选后，为您选定了'和平商务酒店'，价格220元/晚，评分4.2分。支付已成功完成，订单号：ORD123456。祝您入住愉快！"

示例（失败）：
"很抱歉，在中山大道搜索到了3家酒店，但按照您100-150元的预算筛选后，没有符合条件的酒店。建议您适当提高预算或选择其他街道。如需帮助，请随时告诉我。"
"""),
        HumanMessage(content=json.dumps(summary_data, ensure_ascii=False, indent=2))
    ]

    try:
        ai_msg = llm.invoke(prompt)
        final_response = getattr(ai_msg, "content", "总结生成失败")
        state["final_response"] = final_response
        _append_log(state, "[节点7-总结] 已生成最终回复")
    except Exception as e:
        # 生成一个简单的备用回复
        if payment_status == "success":
            state["final_response"] = f"已为您预订{selected_hotel.get('name') if selected_hotel else '酒店'}。{payment_message}"
        elif error_message:
            state["final_response"] = f"很抱歉，{error_message}"
        else:
            state["final_response"] = f"预订未成功。{payment_message if payment_message else '请稍后重试或联系客服。'}"
        _append_log(state, f"[节点7-总结] 使用备用回复（LLM错误: {str(e)}）")

    return state


# ============ 路由函数 ============
def route_after_search(state: BookingState) -> str:
    """搜索后的路由：如果搜索结果为空，直接跳到总结"""
    if not state.get("search_results"):
        return "summarize"
    return "filter_select"


def route_after_payment(state: BookingState) -> str:
    """支付后始终进入总结节点"""
    return "summarize"


# ============ 构建工作流 ============
def build_booking_workflow():
    """构建完整的预订工作流图"""
    graph = StateGraph(BookingState)

    # 添加7个节点
    graph.add_node("intent", intent_understanding_node)          # 节点1：意图理解
    graph.add_node("extract", parameter_extraction_node)         # 节点2：参数提取
    graph.add_node("plan", search_planning_node)                 # 节点3：搜索规划
    graph.add_node("search", hotel_search_node)                  # 节点4：酒店搜索
    graph.add_node("filter_select", filter_and_selection_node)   # 节点5：筛选选择
    graph.add_node("pay", payment_node)                          # 节点6：支付
    graph.add_node("summarize", summarization_node)              # 节点7：总结

    # 设置入口点
    graph.set_entry_point("intent")

    # 构建节点间的连接
    graph.add_edge("intent", "extract")         # 1 -> 2
    graph.add_edge("extract", "plan")           # 2 -> 3
    graph.add_edge("plan", "search")            # 3 -> 4

    # 搜索后的条件路由
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {
            "filter_select": "filter_select",   # 有结果 -> 5
            "summarize": "summarize"            # 无结果 -> 7
        }
    )

    graph.add_edge("filter_select", "pay")      # 5 -> 6

    # 支付后的路由
    graph.add_conditional_edges(
        "pay",
        route_after_payment,
        {"summarize": "summarize"}              # 6 -> 7
    )

    graph.add_edge("summarize", END)            # 7 -> 结束

    return graph.compile()


# ============ 对外接口函数 ============
def run_booking_workflow(user_request: str) -> str:
    """
    运行酒店预订工作流的主函数

    参数：
        user_request: 用户的自然语言请求

    返回：
        str: 最终的回复文本

    使用示例：
        >>> result = run_booking_workflow("请帮我在和平路找一个200到300元的酒店")
        >>> print(result)
    """
    # 构建工作流
    app = build_booking_workflow()

    # 初始化状态
    initial_state = {
        "user_request": user_request,
        "payment_status": "pending",
        "reasoning_log": [],
    }

    try:
        # 执行工作流
        result = app.invoke(initial_state)

        # 返回最终回复
        return result.get("final_response", "抱歉，系统处理出错，请稍后重试。")
    except Exception as e:
        return f"系统错误：{str(e)}。请稍后重试或联系技术支持。"


# ============ 调试辅助函数 ============
def run_booking_workflow_with_details(user_request: str) -> Dict[str, Any]:
    """
    运行工作流并返回详细的执行结果（用于调试）

    参数：
        user_request: 用户的自然语言请求

    返回：
        Dict: 包含最终回复和完整状态的字典
    """
    app = build_booking_workflow()
    initial_state = {
        "user_request": user_request,
        "payment_status": "pending",
        "reasoning_log": [],
    }

    try:
        result = app.invoke(initial_state)
        return {
            "final_response": result.get("final_response", "处理出错"),
            "full_state": result,
            "reasoning_log": result.get("reasoning_log", []),
        }
    except Exception as e:
        return {
            "final_response": f"系统错误：{str(e)}",
            "full_state": None,
            "reasoning_log": [f"错误: {str(e)}"],
        }
