"""
SN noderank 计算工具模块

提供从 SN 图或拓扑文件计算 noderank 的功能。
参考 tester1/dataset_generate.py 的 _compute_sn_noderank 实现。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import networkx as nx

__all__ = [
    "compute_sn_noderank_from_graph",
    "compute_sn_noderank_from_file",
]


def compute_sn_noderank_from_graph(
    G_sn: nx.Graph,
    use_residual_resources: bool = False
) -> np.ndarray:
    """
    从 SN 图计算 noderank。
    
    算法说明：
    1. 计算初始 noderank: H(u) = cpu * comm_bandwidth（或基于剩余资源），然后归一化
    2. 构建无向邻接矩阵
    3. 计算前向概率 pF（基于邻居的 H 值）
    4. 进行两次传播并归一化
    5. 做三次幂并归一化
    
    Args:
        G_sn: SN 网络图（NetworkX Graph 或 DiGraph）
        use_residual_resources: 如果为True，使用剩余资源（cpu_res, mem_res, disk_res）
                               如果为False，使用初始资源（cpu, bandwidth）
    
    Returns:
        noderank: numpy 数组，每个元素对应一个 SN 节点的 noderank 值
        节点顺序与 sorted(G_sn.nodes()) 的顺序一致
    """
    # 转换为无向图
    if G_sn.is_directed():
        G_undirected = G_sn.to_undirected()
    else:
        G_undirected = G_sn
    
    # 获取节点列表（排序以确保顺序一致）
    sn_node_list = sorted(G_undirected.nodes())
    n = len(sn_node_list)
    
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    
    # 创建节点索引映射
    node_to_idx = {node_id: idx for idx, node_id in enumerate(sn_node_list)}
    
    # 步骤1: 计算资源评估 H(u)
    H = np.zeros(n, dtype=np.float64)
    for node_id in sn_node_list:
        node_data = G_undirected.nodes[node_id]
        if use_residual_resources:
            # 使用剩余资源：H(u) = cpu_res * comm_bandwidth
            cpu = float(node_data.get('cpu_res', 0.0))
            comm_bw = float(node_data.get('comm_bandwidth', node_data.get('bandwidth', 0.0)))
        else:
            # 使用初始资源：H(u) = cpu * comm_bandwidth
            cpu = float(node_data.get('cpu', 0.0))
            comm_bw = float(node_data.get('comm_bandwidth', node_data.get('bandwidth', 0.0)))
        idx = node_to_idx[node_id]
        H[idx] = cpu * comm_bw
    
    H_sum = np.sum(H) if np.sum(H) > 0 else 1.0
    NR = H / H_sum  # 初始 noderank
    
    # 步骤2: 构建邻接（无向）
    adjacency: List[List[int]] = [[] for _ in range(n)]
    for u, v in G_undirected.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        if v_idx not in adjacency[u_idx]:
            adjacency[u_idx].append(v_idx)
        if u_idx not in adjacency[v_idx]:
            adjacency[v_idx].append(u_idx)
    
    # 步骤3: 计算前向概率 pF (只在邻居间分布)
    pF = np.zeros((n, n), dtype=np.float64)
    for u_idx in range(n):
        nbrs = adjacency[u_idx]
        if not nbrs:
            continue
        denom = np.sum(H[nbrs])
        if denom <= 0:
            continue
        for v_idx in nbrs:
            pF[u_idx, v_idx] = H[v_idx] / denom
    
    # 步骤4: 进行两次传播并归一化，然后做三次幂再归一化
    PF_U = 0.20
    NR_curr = NR
    for _ in range(2):
        NR_next = NR_curr + PF_U * (pF @ NR_curr)
        s = np.sum(NR_next)
        NR_curr = NR_next / (s if s > 0 else 1.0)
    
    NR_final = NR_curr ** 3
    s2 = np.sum(NR_final)
    NR_final = NR_final / (s2 if s2 > 0 else 1.0)
    
    return NR_final


def compute_sn_noderank_from_file(sn_topology_path: str) -> np.ndarray:
    """
    从 SN 拓扑文件计算 noderank。
    
    Args:
        sn_topology_path: SN 拓扑文件路径（JSON 格式）
    
    Returns:
        noderank: numpy 数组
    """
    with open(sn_topology_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
    
    # 构建 NetworkX 图
    directed = js.get('directed', False)
    G = nx.DiGraph() if directed else nx.Graph()
    
    # 添加节点
    for n in js['nodes']:
        node_id = int(n['id'])
        G.add_node(
            node_id,
            cpu=float(n.get('cpu', 0.0)),
            memory=float(n.get('memory', n.get('memory', 0.0))),
            disk=float(n.get('disk', 0.0)),
            bandwidth=float(n.get('bandwidth', 0.0)),
            comm_bandwidth=float(n.get('comm_bandwidth', n.get('bandwidth', 0.0))),
        )
    
    # 添加边
    for e in js['links']:
        u = int(e['source'])
        v = int(e['target'])
        G.add_edge(
            u, v,
            weight=float(e.get('weight', 1.0)),
            bandwidth=float(e.get('bandwidth', 0.0)),
        )
    
    # 计算 noderank
    return compute_sn_noderank_from_graph(G)

