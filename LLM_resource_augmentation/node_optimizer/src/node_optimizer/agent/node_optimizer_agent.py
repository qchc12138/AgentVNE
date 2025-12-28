"""节点优化智能体 - VN与SN匹配分析系统

这个智能体通过LLM分析虚拟节点(VN)的提示词，判断是否需要特殊执行环境，
并匹配到合适的基础网络节点(SN, Substrate Network)。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# 导入LLM配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import llm


# ============ Pydantic 输出结构定义 ============
class VNAnalysis(BaseModel):
    """单个VN节点的分析结果"""
    vn_id: str = Field(description="虚拟节点ID，如VN1, VN2等")
    vn_name: str = Field(description="虚拟节点名称")
    requires_special_env: bool = Field(description="是否需要特殊执行环境")
    special_env_type: Optional[str] = Field(default=None, description="所需的特殊环境类型")
    reasoning: str = Field(description="判断理由")


class SNSelection(BaseModel):
    """为需要特殊环境的VN选择的SN节点"""
    vn_id: str = Field(description="虚拟节点ID")
    selected_sn_id: int = Field(description="选择的基础网络节点ID（数字）")
    selected_sn_name: str = Field(description="选择的基础网络节点名称")
    match_reasoning: str = Field(description="匹配理由")
    bias: float = Field(ge=0, le=1, description="偏置信息，固定为0.5")


class NodeOptimizerOutput(BaseModel):
    """节点优化器的完整输出结构"""
    workflow_name: str = Field(description="工作流名称")
    total_vn_count: int = Field(description="VN节点总数")
    vn_analysis: List[VNAnalysis] = Field(description="所有VN节点的分析结果")
    special_env_vn_count: int = Field(description="需要特殊环境的VN节点数量")
    sn_selections: List[SNSelection] = Field(description="为需要特殊环境的VN选择的SN节点")
    summary: str = Field(description="总结性描述")


# ============ 数据加载函数 ============
def load_vn_data(file_path: str = None) -> Dict[str, Any]:
    """加载VN节点数据"""
    if file_path is None:
        current_dir = Path(__file__).parent
        file_path = current_dir / "../data/vn_nodes.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_sn_data(file_path: str = None) -> Dict[str, Any]:
    """加载SN节点数据"""
    if file_path is None:
        current_dir = Path(__file__).parent
        file_path = current_dir / "../data/sn_nodes.json"

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============ LLM分析函数 ============
def analyze_vn_with_llm(node: Dict[str, Any], available_env_types: List[str]) -> VNAnalysis:
    """使用LLM分析单个VN节点是否需要特殊环境"""

    vn_id = node["node_id"]
    vn_name = node["node_name"]
    system_prompt = node.get("system_prompt", "")
    user_prompt_template = node.get("user_prompt_template", "")

    # 构建提示词给LLM分析
    analysis_prompt = f"""你是一个节点分析专家。请分析以下虚拟节点(VN)的提示词，判断它是否需要特殊的执行环境。

**虚拟节点信息：**
- 节点ID: {vn_id}
- 节点名称: {vn_name}
- 系统提示词: {system_prompt if system_prompt else "无"}
- 用户提示词模板: {user_prompt_template if user_prompt_template else "无"}

**可用的特殊环境类型：**
{', '.join(available_env_types) if available_env_types else "无特殊环境"}

**分析要求：**
1. 仔细阅读节点的提示词内容
2. 判断该节点是否需要特殊的执行环境（如安全支付环境、摄像头硬件等）
3. 如果需要，指出需要哪种类型的特殊环境
4. 给出详细的判断理由

请以JSON格式返回分析结果：
```json
{{
    "requires_special_env": true/false,
    "special_env_type": "环境类型名称或null",
    "reasoning": "详细的判断理由"
}}
```"""

    try:
        messages = [
            SystemMessage(content="你是一个专业的节点分析专家，擅长分析系统提示词和判断执行环境需求。"),
            HumanMessage(content=analysis_prompt)
        ]

        response = llm.invoke(messages)
        content = response.content

        # 解析JSON响应
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            # 尝试直接解析
            result = json.loads(content)

        return VNAnalysis(
            vn_id=vn_id,
            vn_name=vn_name,
            requires_special_env=result.get("requires_special_env", False),
            special_env_type=result.get("special_env_type"),
            reasoning=result.get("reasoning", "LLM分析结果")
        )

    except Exception as e:
        # LLM调用失败的后备方案
        return VNAnalysis(
            vn_id=vn_id,
            vn_name=vn_name,
            requires_special_env=False,
            special_env_type=None,
            reasoning=f"分析过程出错：{str(e)}"
        )


def select_sn_for_vn(vn_analysis: VNAnalysis, sn_data: Dict[str, Any]) -> Optional[SNSelection]:
    """为需要特殊环境的VN选择合适的SN节点"""
    if not vn_analysis.requires_special_env:
        return None

    required_env = vn_analysis.special_env_type

    # 查找能提供该环境的SN节点
    special_nodes = sn_data.get("special_environment_nodes", [])

    for sn in special_nodes:
        if sn.get("special_environment_type") == required_env:
            return SNSelection(
                vn_id=vn_analysis.vn_id,
                selected_sn_id=sn["node_id"],
                selected_sn_name=sn["node_name"],
                match_reasoning=f"{sn['node_name']}提供{required_env}环境。{sn['description']}",
                bias=0.5
            )

    # 未找到匹配的SN
    return SNSelection(
        vn_id=vn_analysis.vn_id,
        selected_sn_id=0,
        selected_sn_name="未找到匹配节点",
        match_reasoning=f"未找到能提供{required_env}环境的基础网络节点",
        bias=0.5
    )


def generate_summary_with_llm(vn_analyses: List[VNAnalysis],
                              sn_selections: List[SNSelection],
                              workflow_name: str) -> str:
    """使用LLM生成详细的用户总结"""

    # 构建分析结果描述
    vn_details = []
    for vn in vn_analyses:
        detail = f"- {vn.vn_id}({vn.vn_name}): "
        if vn.requires_special_env:
            detail += f"需要{vn.special_env_type}环境。理由：{vn.reasoning}"
        else:
            detail += f"无需特殊环境。理由：{vn.reasoning}"
        vn_details.append(detail)

    sn_details = []
    for sn in sn_selections:
        detail = f"- {sn.vn_id}匹配到{sn.selected_sn_name}(ID:{sn.selected_sn_id})。{sn.match_reasoning}"
        sn_details.append(detail)

    summary_prompt = f"""请为用户生成一个清晰、专业的分析总结。

**工作流名称：** {workflow_name}
**VN节点总数：** {len(vn_analyses)}
**需要特殊环境的VN：** {len(sn_selections)}

**VN节点分析详情：**
{chr(10).join(vn_details)}

**SN匹配结果：**
{chr(10).join(sn_details) if sn_details else "无需匹配"}

**要求：**
1. 用通俗易懂的语言总结分析结果
2. 重点说明哪些VN需要特殊环境，为什么需要（基于提示词判断）
3. 说明匹配到了哪些SN节点，为什么选择它们
4. 语言要专业但友好，100-200字
5. 直接输出总结内容，不要有任何前缀或后缀"""

    try:
        messages = [
            SystemMessage(content="你是一个专业的技术分析师，擅长用清晰的语言解释技术分析结果。"),
            HumanMessage(content=summary_prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        # LLM调用失败的后备总结
        if len(sn_selections) == 0:
            return f"分析完成。工作流'{workflow_name}'包含{len(vn_analyses)}个虚拟节点，所有节点均可在普通计算环境中运行。"
        else:
            parts = [f"分析完成。工作流'{workflow_name}'包含{len(vn_analyses)}个虚拟节点，其中{len(sn_selections)}个需要特殊环境："]
            for sn in sn_selections:
                parts.append(f"{sn.vn_id}匹配到{sn.selected_sn_name}(ID:{sn.selected_sn_id})。")
            return " ".join(parts)


# ============ 主分析流程 ============
def run_node_optimizer(vn_file: str = None, sn_file: str = None) -> NodeOptimizerOutput:
    """
    运行节点优化器主流程

    Args:
        vn_file: VN节点数据文件路径（可选）
        sn_file: SN节点数据文件路径（可选）

    Returns:
        NodeOptimizerOutput: 结构化的分析结果
    """
    # 加载数据
    vn_data = load_vn_data(vn_file)
    sn_data = load_sn_data(sn_file)

    # 获取可用的特殊环境类型
    available_env_types = [
        sn.get("special_environment_type")
        for sn in sn_data.get("special_environment_nodes", [])
    ]

    # 使用LLM分析每个VN节点
    vn_analyses = []
    for node in vn_data.get("nodes", []):
        analysis = analyze_vn_with_llm(node, available_env_types)
        vn_analyses.append(analysis)

    # 为需要特殊环境的VN选择SN
    sn_selections = []
    for vn_analysis in vn_analyses:
        if vn_analysis.requires_special_env:
            selection = select_sn_for_vn(vn_analysis, sn_data)
            if selection:
                sn_selections.append(selection)

    # 统计信息
    special_env_count = len(sn_selections)

    # 使用LLM生成总结
    summary = generate_summary_with_llm(
        vn_analyses,
        sn_selections,
        vn_data["workflow_name"]
    )

    # 构建输出
    output = NodeOptimizerOutput(
        workflow_name=vn_data["workflow_name"],
        total_vn_count=len(vn_analyses),
        vn_analysis=vn_analyses,
        special_env_vn_count=special_env_count,
        sn_selections=sn_selections,
        summary=summary
    )

    return output


# ============ 结果保存函数 ============
def save_results(output: NodeOptimizerOutput, output_file: str = None) -> str:
    """
    保存分析结果到JSON文件

    Args:
        output: 分析结果
        output_file: 输出文件路径（可选）

    Returns:
        str: 输出文件路径
    """
    if output_file is None:
        current_dir = Path(__file__).parent
        results_dir = current_dir / "../results"
        results_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"node_optimizer_result_{timestamp}.json"

    # 转换为字典并保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output.model_dump(), f, ensure_ascii=False, indent=2)

    return str(output_file)
