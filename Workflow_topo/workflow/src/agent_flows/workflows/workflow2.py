"""LangGraph工作流：摄像头图像处理助手（workflow2）

该工作流包含6个节点，模拟摄像头采集、预处理、特征提取和需求响应的全过程。
"""

import json
import re
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..common.utils import llm
from ..common.workflow2_tools import (
    capture_camera_image,
    build_congestion_narrative,
    extract_image_features,
    estimate_congestion_level,
    preprocess_image,
    rate_noise_level,
    summarize_scene_from_features,
)


FIXED_SCENE = "固定场景-城市路口交通监控"


class CameraWorkflowState(TypedDict, total=False):
    """工作流状态"""

    user_request: str
    intent: str
    camera_id: str
    scene: str
    capture_info: Dict[str, Any]
    preprocess_steps: List[str]
    features: Dict[str, Any]
    analysis: str
    final_response: str
    error_message: str
    reasoning_log: List[str]


def _append_log(state: CameraWorkflowState, note: str) -> None:
    log = state.get("reasoning_log", [])
    log.append(note)
    state["reasoning_log"] = log


def intent_understanding_node(state: CameraWorkflowState) -> CameraWorkflowState:
    user_request = state.get("user_request", "")
    prompt = [
        SystemMessage(
            content="""你是图像处理意图理解智能体。
请用一句话概括用户想对摄像头画面做什么，语气简洁。"""
        ),
        HumanMessage(content=user_request),
    ]
    try:
        ai_msg = llm.invoke(prompt)
        intent = getattr(ai_msg, "content", "未能识别意图")
        state["intent"] = intent
        state["scene"] = FIXED_SCENE
        state["camera_id"] = state.get("camera_id") or "cam-fixed-01"
        _append_log(state, f"[节点1-意图] {intent}｜场景固定为{FIXED_SCENE}")
    except Exception as e:
        state["intent"] = "意图识别失败"
        state["error_message"] = str(e)
        _append_log(state, f"[节点1-意图] 错误: {e}")
    return state


def camera_capture_node(state: CameraWorkflowState) -> CameraWorkflowState:
    if state.get("error_message"):
        return state

    camera_id = state.get("camera_id", "cam-fixed-01")
    scene = FIXED_SCENE

    try:
        capture_info = capture_camera_image(camera_id=camera_id, scene=scene, secure_environment=True)
        state["capture_info"] = capture_info
        _append_log(state, f"[节点2-采集] 成功捕获场景: {scene} ({capture_info.get('path')})")
    except Exception as e:
        state["error_message"] = f"采集失败: {e}"
        _append_log(state, f"[节点2-采集] 错误: {e}")
    return state


def preprocessing_node(state: CameraWorkflowState) -> CameraWorkflowState:
    if state.get("error_message"):
        return state

    capture_info = state.get("capture_info")
    if not capture_info:
        state["error_message"] = "没有捕获到图像，无法预处理"
        _append_log(state, "[节点3-预处理] 无图像")
        return state

    image = capture_info.get("image")
    if image is None:
        state["error_message"] = "图像数据缺失"
        _append_log(state, "[节点3-预处理] 图像数据缺失")
        return state

    user_request = state.get("user_request", "")
    ops = {
        "grayscale": True,
        "gaussian_blur": True,
        "denoise": bool(re.search(r"去噪|降噪|滤波", user_request)),
    }

    try:
        processed, steps = preprocess_image(image, operations=ops)
        state["capture_info"]["image"] = processed
        state["preprocess_steps"] = steps
        if any(s.startswith("gpu_smooth") or s.startswith("gpu_skip") for s in steps):
            state["gpu_used"] = True
            _append_log(state, f"[节点3-预处理] GPU路径: {steps}")
        _append_log(state, f"[节点3-预处理] 步骤: {steps}")
    except Exception as e:
        state["error_message"] = f"预处理失败: {e}"
        _append_log(state, f"[节点3-预处理] 错误: {e}")
    return state


def feature_extraction_node(state: CameraWorkflowState) -> CameraWorkflowState:
    if state.get("error_message"):
        return state

    capture_info = state.get("capture_info")
    image = capture_info.get("image") if capture_info else None
    if image is None:
        state["error_message"] = "缺少图像，无法提取特征"
        _append_log(state, "[节点4-特征] 无图像")
        return state

    try:
        features = extract_image_features(image)
        state["features"] = features
        _append_log(state, "[节点4-特征] 已提取基本特征")
    except Exception as e:
        state["error_message"] = f"特征提取失败: {e}"
        _append_log(state, f"[节点4-特征] 错误: {e}")
    return state


def task_execution_node(state: CameraWorkflowState) -> CameraWorkflowState:
    if state.get("error_message"):
        return state

    user_request = state.get("user_request", "")
    intent = state.get("intent", "")
    features = state.get("features", {})
    scene_desc = summarize_scene_from_features(features) if features else "暂无特征"
    congestion = estimate_congestion_level(features) if features else {"level": "未知", "reason": "无特征"}
    noise = rate_noise_level(features) if features else {"level": "未知", "note": "无特征"}
    narrative_hint = build_congestion_narrative(features) if features else "未能获得有效画面特征"

    prompt = [
        SystemMessage(
            content="""你是图像分析助手。
请输出一段连续、自然、口语化的中文描述（70-140字，单段）。
要求：
1) 明确是否拥堵，引用边缘密度/占用度；若提到“综合得分”，需解释它是两项的加权分，分高意味着更拥堵；
2) 简述画质/噪声感受；
3) 单轮回答，不要邀请反馈，不要列点，不要给出建议。"""
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "用户需求": user_request,
                    "系统理解": intent,
                    "场景特征": scene_desc,
                    "特征明细": features,
                    "拥堵估计": congestion,
                    "画质评估": noise,
                    "叙述线索": narrative_hint,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    ]
    try:
        ai_msg = llm.invoke(prompt)
        analysis = getattr(ai_msg, "content", "分析失败")
        state["analysis"] = analysis
        _append_log(state, "[节点5-分析] 已生成分析")
    except Exception as e:
        fallback = (
            f"画质{noise.get('level')}，{congestion.get('reason')}。"
            f"当前判断：{congestion.get('level')}。"
        )
        state["analysis"] = fallback
        _append_log(state, f"[节点5-分析] 备用分析（LLM失败: {e}）")
    return state


def summarization_node(state: CameraWorkflowState) -> CameraWorkflowState:
    user_request = state.get("user_request", "")
    scene = FIXED_SCENE
    preprocess_steps = state.get("preprocess_steps", [])
    analysis = state.get("analysis", "")
    error_message = state.get("error_message", "")
    features = state.get("features", {})
    congestion = estimate_congestion_level(features) if features else None
    noise = rate_noise_level(features) if features else None

    prompt = [
        SystemMessage(
            content="""你是总结助手。
请用一段友好、自然的中文（不分行，最多160字）总结：
1) 点出场景并说明这是单轮回应；
2) 概述预处理；
3) 说明拥堵判定并引用边缘密度/占用度，若提到“综合得分”须解释它是两项加权分、高代表更拥堵；
4) 简述画质/噪声感受；
5) 不要提供建议，不要邀请反馈。"""
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "用户需求": user_request,
                    "场景": scene,
                    "预处理": preprocess_steps,
                    "分析": analysis,
                    "拥堵估计": congestion,
                    "画质评估": noise,
                    "错误": error_message,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    ]

    try:
        ai_msg = llm.invoke(prompt)
        final_response = getattr(ai_msg, "content", "未能生成总结")
        state["final_response"] = final_response
        _append_log(state, "[节点6-总结] 完成")
    except Exception as e:
        state["final_response"] = (
            f"本次处理遇到问题：{error_message or str(e)}。请稍后重试或更换图像。"
        )
        _append_log(state, f"[节点6-总结] 备用总结，原因: {e}")
    return state


def build_camera_workflow():
    graph = StateGraph(CameraWorkflowState)

    graph.add_node("intent", intent_understanding_node)
    graph.add_node("capture", camera_capture_node)
    graph.add_node("preprocess", preprocessing_node)
    graph.add_node("features", feature_extraction_node)
    graph.add_node("analyze", task_execution_node)
    graph.add_node("summarize", summarization_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "capture")
    graph.add_edge("capture", "preprocess")
    graph.add_edge("preprocess", "features")
    graph.add_edge("features", "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def run_camera_workflow(user_request: str) -> str:
    app = build_camera_workflow()
    initial_state: CameraWorkflowState = {
        "user_request": user_request,
        "reasoning_log": [],
    }
    result = app.invoke(initial_state)
    return result.get("final_response", "处理失败，请稍后重试。")


def run_camera_workflow_with_details(user_request: str) -> Dict[str, Any]:
    app = build_camera_workflow()
    initial_state: CameraWorkflowState = {
        "user_request": user_request,
        "reasoning_log": [],
    }
    result = app.invoke(initial_state)
    return {
        "final_response": result.get("final_response", "处理失败"),
        "full_state": result,
        "reasoning_log": result.get("reasoning_log", []),
    }
