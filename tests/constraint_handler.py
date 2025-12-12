"""
约束节点处理工具模块

提供约束节点分离、资源检查和放置等功能，用于支持两阶段放置策略。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

import os
import sys
from pathlib import Path

#region sys.path 管理
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
#endregion

from env import SimuVNEEnv

__all__ = [
    "separate_constraint_nodes",
    "place_constraint_nodes",
    "check_constraint_node_resources",
]


def separate_constraint_nodes(vn: Data) -> Tuple[List[int], List[int], Dict[int, int]]:
    """
    分离约束节点和非约束节点。
    
    Args:
        vn: VN 图数据对象，可能包含 constraint_nodes 属性
    
    Returns:
        (non_constraint_vn_indices, constraint_vn_indices, constraint_mapping)
        - non_constraint_vn_indices: 非约束节点的 VN 索引列表
        - constraint_vn_indices: 约束节点的 VN 索引列表
        - constraint_mapping: {vn_idx: sn_node_id} 约束节点映射字典
    """
    non_constraint_indices: List[int] = []
    constraint_indices: List[int] = []
    constraint_mapping: Dict[int, int] = {}
    
    # 检查是否有 constraint_nodes 属性
    if not hasattr(vn, 'constraint_nodes') or vn.constraint_nodes is None:
        # 所有节点都是非约束节点
        num_nodes = vn.x.size(0)
        return list(range(num_nodes)), [], {}
    
    # 分离节点
    constraint_nodes_list = vn.constraint_nodes
    for vn_idx in range(vn.x.size(0)):
        if vn_idx < len(constraint_nodes_list):
            constraint_sn_id = constraint_nodes_list[vn_idx]
            if constraint_sn_id is None:
                non_constraint_indices.append(vn_idx)
            else:
                constraint_indices.append(vn_idx)
                constraint_mapping[vn_idx] = int(constraint_sn_id)
        else:
            # 如果 constraint_nodes 列表长度不足，视为非约束节点
            non_constraint_indices.append(vn_idx)
    
    return non_constraint_indices, constraint_indices, constraint_mapping


def check_constraint_node_resources(
    env: SimuVNEEnv,
    vn: Data,
    vn_idx: int,
    sn_node_id: int
) -> Tuple[bool, Optional[str]]:
    """
    检查约束节点是否可以放置到指定 SN 节点。
    
    Args:
        env: 环境对象（可以是副本）
        vn: VN 图数据对象
        vn_idx: VN 节点索引
        sn_node_id: SN 节点 ID
    
    Returns:
        (can_place, reason)
        - can_place: 是否可以放置
        - reason: 不能放置的原因（如果可以放置则为 None）
    """
    # 检查 SN 节点是否存在
    if sn_node_id not in env.G_sn.nodes:
        return False, f"SN节点 {sn_node_id} 不存在"
    
    # 获取 VN 节点需求（归一化）
    feats = vn.x[vn_idx]
    norm_cpu = float(feats[0].item())
    norm_mem = float(feats[1].item())
    norm_disk = float(feats[2].item())
    
    # 转为绝对需求
    abs_cpu = norm_cpu * (env._sn_max_capacity['cpu_max'] + 1e-8)
    abs_mem = norm_mem * (env._sn_max_capacity['mem_max'] + 1e-8)
    abs_disk = norm_disk * (env._sn_max_capacity['disk_max'] + 1e-8)
    
    # 获取 SN 节点剩余资源
    sn_node = env.G_sn.nodes[sn_node_id]
    res_cpu = float(sn_node.get('cpu_res', 0.0))
    res_mem = float(sn_node.get('mem_res', 0.0))
    res_disk = float(sn_node.get('disk_res', 0.0))
    
    # 检查资源是否足够
    if abs_cpu > res_cpu + 1e-9:
        return False, f"CPU资源不足: 需要 {abs_cpu:.3f}, 剩余 {res_cpu:.3f}"
    if abs_mem > res_mem + 1e-9:
        return False, f"内存资源不足: 需要 {abs_mem:.3f}, 剩余 {res_mem:.3f}"
    if abs_disk > res_disk + 1e-9:
        return False, f"磁盘资源不足: 需要 {abs_disk:.3f}, 剩余 {res_disk:.3f}"
    
    return True, None


def place_constraint_nodes(
    env: SimuVNEEnv,
    vn: Data,
    non_constraint_mapping: Dict[int, int],
    constraint_mapping: Dict[int, int]
) -> Tuple[bool, Dict[int, int], Optional[str]]:
    """
    在非约束节点映射的基础上，放置约束节点。
    
    此函数会在环境副本上检查资源并应用映射，但不进行实际的资源扣减。
    资源扣减应该由调用者通过 env._apply_mapping 统一处理。
    
    Args:
        env: 环境对象（副本，用于资源检查）
        vn: VN 图数据对象
        non_constraint_mapping: 非约束节点的映射 {vn_idx: sn_node_id}
        constraint_mapping: 约束节点的映射 {vn_idx: sn_node_id}
    
    Returns:
        (success, full_mapping, failure_reason)
        - success: 是否成功放置所有约束节点
        - full_mapping: 完整映射（包含非约束和约束节点）
        - failure_reason: 失败原因（如果失败）
    """
    # 构建完整映射
    full_mapping = dict(non_constraint_mapping)
    
    # 检查并添加约束节点映射
    for vn_idx, sn_node_id in constraint_mapping.items():
        # 检查资源
        can_place, reason = check_constraint_node_resources(
            env, vn, vn_idx, sn_node_id
        )
        if not can_place:
            return False, {}, f"约束节点 VN{vn_idx} 无法放置到 SN{sn_node_id}: {reason}"
        
        # 添加到映射
        full_mapping[vn_idx] = sn_node_id
    
    return True, full_mapping, None

