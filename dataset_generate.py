#!/usr/bin/env python3
"""
预训练数据集生成脚本
基于底层网络与 workflow 的拓扑，按要求生成 <x, y> 预训练数据集：
- 在底层网络上依次放置 10 个相同的 workflow；
- 每放置一个 workflow 前，记录当前 workflow 拓扑与当前底层网络状态，作为 x；
- 用参考的 NodeRank 计算（底层网络来自 calculate_noderank_2 的公式，workflow 的节点 rank 直接读取）
  构造分配概率矩阵 y（将当前 SN 的 noderank 重复 N1 行得到 N1×N2）。
- 以贪心方式将该 workflow 的节点映射到底层网络（按各自 noderank 从高到低，检查 cpu/memory/disk 约束），
  并更新底层网络资源；
- 重复上述过程，直到 10 个 workflow 放置完成；将整个过程重复 50 次。
"""

import torch
import os
import json
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from torch_geometric.data import Data
from tqdm import tqdm


def _load_json(path: str) -> Dict:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_edge_index(links: List[Dict], directed: bool) -> torch.Tensor:
    """根据 links 构造符合 torch_geometric 要求的 edge_index [2, E] (long)。
    若无向，则为每条边加入双向条目。
    """
    edges: List[Tuple[int, int]] = []
    for link in links:
        u = int(link['source'])
        v = int(link['target'])
        edges.append((u, v))
        if not directed:
            edges.append((v, u))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def _nodes_to_features(nodes: List[Dict], is_workflow: bool = False, sn_max_capacity: Dict[str, float] = None) -> torch.Tensor:
    """将节点属性映射为 6 维特征并归一化。
    
    特征顺序: [cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]
    
    Args:
        nodes: 节点列表
        is_workflow: 是否为 workflow 节点（True=VN需求，False=SN容量）
        sn_max_capacity: SN最大容量字典，用于归一化
    
    Returns:
        归一化后的特征张量 [N, 6]
    """
    if sn_max_capacity is None:
        sn_max_capacity = {
            'cpu_max': 4.0,
            'mem_max': 4.0,
            'disk_max': 6.0,
            'bw_max': 10.0,
            'comm_bw_max': 10.0,
        }
    
    feats: List[List[float]] = []
    for n in nodes:
        cpu = float(n.get('cpu', 0.0))
        memory = float(n.get('memory', 0.0))
        disk = float(n.get('disk', 0.0))
        bandwidth = float(n.get('bandwidth', 1.0))
        comm_bw = float(n.get('comm_bandwidth', bandwidth))
        
        # 归一化：除以 SN 最大容量
        cpu_norm = cpu / (sn_max_capacity['cpu_max'] + 1e-8)
        mem_norm = memory / (sn_max_capacity['mem_max'] + 1e-8)
        disk_norm = disk / (sn_max_capacity['disk_max'] + 1e-8)
        bw_norm = bandwidth / (sn_max_capacity['bw_max'] + 1e-8)
        comm_bw_norm = comm_bw / (sn_max_capacity['comm_bw_max'] + 1e-8)
        
        # 特征顺序: [cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]
        feats.append([cpu_norm, mem_norm, disk_norm, bw_norm, comm_bw_norm, 0.0])
    
    return torch.tensor(feats, dtype=torch.float)


def _topology_to_pyg_data(
    topo: Dict,
    is_workflow: bool = False,
    sn_max_capacity: Dict[str, float] = None
) -> Data:
    """将拓扑 dict 转为 torch_geometric.data.Data（带归一化）。
    需要字段: topo['nodes'] (带资源/需求), topo['links'] (source/target)。
    使用 topo['directed'] 判断是否有向。
    
    注意：保持原始节点顺序，不进行排序。links中的source/target应为节点索引。
    
    Args:
        topo: 拓扑字典
        is_workflow: 是否为 workflow 图
        sn_max_capacity: SN最大容量字典，用于归一化
    
    Returns:
        Data对象，节点顺序与topo['nodes']一致
    """
    nodes = topo['nodes']
    links = topo.get('links', [])
    directed = bool(topo.get('directed', False))
    
    # 直接使用原始节点顺序，无需排序
    x = _nodes_to_features(nodes, is_workflow=is_workflow, sn_max_capacity=sn_max_capacity)
    edge_index = _build_edge_index(links, directed=directed)
    data_obj = Data(x=x, edge_index=edge_index)
    
    return data_obj


def _compute_sn_max_capacity(nodes: List[Dict]) -> Dict[str, float]:
    """计算SN网络的最大容量（用于归一化）
    
    Args:
        nodes: SN节点列表
    
    Returns:
        最大容量字典
    """
    max_cpu = 0.0
    max_mem = 0.0
    max_disk = 0.0
    max_bw = 0.0
    max_comm_bw = 0.0
    
    for n in nodes:
        cpu = float(n.get('cpu', 0.0))
        mem = float(n.get('memory', 0.0))
        disk = float(n.get('disk', 0.0))
        bw = float(n.get('bandwidth', 0.0))
        comm_bw = float(n.get('comm_bandwidth', bw))
        
        max_cpu = max(max_cpu, cpu)
        max_mem = max(max_mem, mem)
        max_disk = max(max_disk, disk)
        max_bw = max(max_bw, bw)
        max_comm_bw = max(max_comm_bw, comm_bw)
    
    return {
        'cpu_max': max_cpu if max_cpu > 0 else 1.0,
        'mem_max': max_mem if max_mem > 0 else 1.0,
        'disk_max': max_disk if max_disk > 0 else 1.0,
        'bw_max': max_bw if max_bw > 0 else 1.0,
        'comm_bw_max': max_comm_bw if max_comm_bw > 0 else 1.0,
    }


def _compute_sn_noderank(nodes: List[Dict], links: List[Dict], directed: bool = False) -> np.ndarray:
    """参考 calculate_noderank_2.py 的逻辑，计算底层网络节点的 NodeRank。
    采用 H(u)=cpu*comm_bandwidth，基于无向邻接构造前向概率 pF，做两轮邻居传播后做 3 次幂并归一化。
    返回 shape=(N2,) 的 numpy 数组。
    """
    n = len(nodes)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    # 资源评估 H(u) = cpu * comm_bandwidth
    H = np.zeros(n, dtype=np.float64)
    for i, node in enumerate(nodes):
        cpu = float(node.get('cpu', 0.0))
        comm_bw = float(node.get('comm_bandwidth', node.get('bandwidth', 0.0)))
        H[i] = cpu * comm_bw
    H_sum = np.sum(H) if np.sum(H) > 0 else 1.0
    NR = H / H_sum  # 初始 noderank

    # 构建邻接（无向）
    adjacency: List[List[int]] = [[] for _ in range(n)]
    for link in links:
        u = int(link['source'])
        v = int(link['target'])
        if v not in adjacency[u]:
            adjacency[u].append(v)
        if u not in adjacency[v]:
            adjacency[v].append(u)

    # 前向概率 pF (只在邻居间分布)
    pF = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        nbrs = adjacency[u]
        if not nbrs:
            continue
        denom = np.sum(H[nbrs])
        if denom <= 0:
            continue
        for v in nbrs:
            pF[u, v] = H[v] / denom

    # 进行两次传播并归一化，然后做三次幂再归一化
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


def _get_vn_neighbors_from_topo(workflow_topo: Dict) -> Dict[int, Set[int]]:
    """从 workflow 拓扑获取 VN 节点的邻居关系（无向，即使边是有向的）"""
    nodes = workflow_topo['nodes']
    N1 = len(nodes)
    neighbors = {i: set() for i in range(N1)}
    
    # 创建节点 ID 到索引的映射
    node_id_to_idx = {}
    for idx, node in enumerate(nodes):
        node_id = int(node.get('id', idx))
        node_id_to_idx[node_id] = idx
    
    links = workflow_topo.get('links', [])
    for link in links:
        u_id = int(link['source'])
        v_id = int(link['target'])
        u_idx = node_id_to_idx.get(u_id)
        v_idx = node_id_to_idx.get(v_id)
        if u_idx is not None and v_idx is not None:
            neighbors[u_idx].add(v_idx)
            neighbors[v_idx].add(u_idx)  # 即使边是有向的，也互为邻居
    
    return neighbors


def _get_sn_k_hop_neighbors_from_topo(sn_topo: Dict, sn_node_id: int, k: int) -> Set[int]:
    """从 SN 拓扑获取节点的 k 跳邻居（包括 k 跳内的所有节点）"""
    if k == 0:
        return {sn_node_id}
    
    # 构建 NetworkX 图
    G = nx.Graph()
    for node in sn_topo['nodes']:
        node_id = int(node.get('id', len(G.nodes)))
        G.add_node(node_id)
    
    for link in sn_topo.get('links', []):
        u = int(link['source'])
        v = int(link['target'])
        G.add_edge(u, v)
    
    # 计算 k 跳邻居
    try:
        paths = nx.single_source_shortest_path_length(G, sn_node_id, cutoff=k)
        return set(paths.keys())
    except:
        return {sn_node_id}


def _check_sn_resource_for_placement(
    sn_nodes: List[Dict],
    sn_node_idx: int,
    wf_nodes: List[Dict],
    wf_node_idx: int,
    temp_mapping: Optional[Dict[int, int]] = None
) -> bool:
    """检查 SN 节点是否有足够资源放置 VN 节点"""
    sn_node = sn_nodes[sn_node_idx]
    wf_node = wf_nodes[wf_node_idx]
    
    # 计算绝对资源需求
    demand_cpu = float(wf_node.get('cpu', 0.0))
    demand_mem = float(wf_node.get('memory', 0.0))
    demand_disk = float(wf_node.get('disk', 0.0))
    
    # 计算当前 SN 节点的可用资源（考虑临时映射中已放置的节点）
    available_cpu = float(sn_node.get('cpu', 0.0))
    available_mem = float(sn_node.get('memory', 0.0))
    available_disk = float(sn_node.get('disk', 0.0))
    
    if temp_mapping:
        # 虚拟扣减当前轮已放置在该SN节点上的VN节点的资源
        for vn_idx, sn_idx in temp_mapping.items():
            if sn_idx == sn_node_idx:
                vn_node_temp = wf_nodes[vn_idx]
                available_cpu -= float(vn_node_temp.get('cpu', 0.0))
                available_mem -= float(vn_node_temp.get('memory', 0.0))
                available_disk -= float(vn_node_temp.get('disk', 0.0))
    
    # 检查剩余资源
    if demand_cpu > available_cpu + 1e-9:
        return False
    if demand_mem > available_mem + 1e-9:
        return False
    if demand_disk > available_disk + 1e-9:
        return False
    return True


def _deduct_resource(
    sn_nodes: List[Dict],
    sn_node_idx: int,
    wf_nodes: List[Dict],
    wf_node_idx: int
) -> None:
    """扣减 SN 节点的资源"""
    sn_node = sn_nodes[sn_node_idx]
    wf_node = wf_nodes[wf_node_idx]
    
    demand_cpu = float(wf_node.get('cpu', 0.0))
    demand_mem = float(wf_node.get('memory', 0.0))
    demand_disk = float(wf_node.get('disk', 0.0))
    
    sn_node['cpu'] = float(sn_node.get('cpu', 0.0)) - demand_cpu
    sn_node['memory'] = float(sn_node.get('memory', 0.0)) - demand_mem
    sn_node['disk'] = float(sn_node.get('disk', 0.0)) - demand_disk


def _rollback_resource_deductions(
    sn_nodes: List[Dict],
    wf_nodes: List[Dict],
    deduction_history: List[Tuple[int, int]]
) -> None:
    """回滚资源扣减
    
    Args:
        sn_nodes: SN节点列表（会被原地修改）
        wf_nodes: VN节点列表
        deduction_history: [(sn_node_idx, wf_node_idx), ...] 资源扣减历史记录
    """
    for sn_node_idx, wf_node_idx in deduction_history:
        sn_node = sn_nodes[sn_node_idx]
        wf_node = wf_nodes[wf_node_idx]
        
        # 恢复资源
        restore_cpu = float(wf_node.get('cpu', 0.0))
        restore_mem = float(wf_node.get('memory', 0.0))
        restore_disk = float(wf_node.get('disk', 0.0))
        
        sn_node['cpu'] = float(sn_node.get('cpu', 0.0)) + restore_cpu
        sn_node['memory'] = float(sn_node.get('memory', 0.0)) + restore_mem
        sn_node['disk'] = float(sn_node.get('disk', 0.0)) + restore_disk


def _greedy_place_workflow(
    workflow_topo: Dict,
    sn_topo: Dict,
    sn_nodes: List[Dict],
    sn_noderank: np.ndarray,
    workflow_noderank: List[float],
    k_hop: int = 1,
    return_placement_order: bool = False
) -> Tuple[Dict[int, int], List[Tuple[int, int, Dict[str, float]]]]:
    """
    使用 BFS 扩展策略将一个 workflow 的所有节点贪心放置到底层网络。
    参考 test.py 的 place_with_bfs_strategy 逻辑。
    
    放置规则：
    1. 选择资源占用最大的 VN 节点作为第一个
    2. 基于 VN 图的 BFS 扩展，优先放在同一 SN 节点上
    3. 如果资源不足，在 k 跳邻居中查找
    4. 按 NodeRank 排序确定优先级
    
    Args:
        workflow_topo: Workflow 拓扑字典
        sn_topo: SN 拓扑字典
        sn_nodes: SN 节点列表（会被原地修改）
        sn_noderank: SN 节点 NodeRank 数组
        workflow_noderank: Workflow 节点 NodeRank 列表
        k_hop: k 跳邻居参数（默认 1）
        return_placement_order: 是否返回放置顺序和资源状态
    
    Returns:
        mapping: VN节点索引到SN节点索引的映射字典
        placement_order: 放置顺序列表，每个元素为 (vn_idx, sn_idx, sn_resources_after)
                        其中 sn_resources_after 是放置后的 SN 节点资源字典
    """
    wf_nodes = workflow_topo['nodes']
    N1 = len(wf_nodes)
    N2 = len(sn_nodes)
    
    # 1. 生成优先级列表（基于 NodeRank，按降序排序）
    # 对每个 VN 节点，SN 节点的优先级列表就是按 sn_noderank 降序排序
    sn_order = list(sorted(range(N2), key=lambda j: sn_noderank[j], reverse=True))
    priority_lists = [sn_order.copy() for _ in range(N1)]  # 每个 VN 节点使用相同的优先级列表
    
    # 2. 获取 VN 邻居关系
    vn_neighbors = _get_vn_neighbors_from_topo(workflow_topo)
    
    # 3. 计算 VN 节点度
    vn_degrees = {i: len(vn_neighbors[i]) for i in range(N1)}
    
    # 4. 计算 VN 节点资源需求（用于排序）
    vn_resource_demands = {}
    for i in range(N1):
        node = wf_nodes[i]
        vn_resource_demands[i] = float(node.get('cpu', 0.0)) + float(node.get('memory', 0.0)) + float(node.get('disk', 0.0))
    
    # 5. 获取 SN 节点 ID 到索引的映射
    sn_id_to_idx = {}
    for idx, node in enumerate(sn_nodes):
        sn_id = int(node.get('id', idx))
        sn_id_to_idx[sn_id] = idx
    
    # 6. 选择第一个 VN 节点（资源占用最大）
    first_vn = max(range(N1), key=lambda i: vn_resource_demands[i])
    
    # 7. 放置第一个 VN 节点（遍历优先级列表，找到第一个能放置的）
    mapping: Dict[int, int] = {}  # VN节点索引 -> SN节点索引
    placement_order: List[Tuple[int, int, Dict[str, float]]] = []  # 放置顺序记录
    resource_deduction_history: List[Tuple[int, int]] = []  # 资源扣减历史记录
    
    first_placed = False
    for first_sn_idx in priority_lists[first_vn]:
        if _check_sn_resource_for_placement(sn_nodes, first_sn_idx, wf_nodes, first_vn, temp_mapping=None):
            # 资源足够，立即扣减
            _deduct_resource(sn_nodes, first_sn_idx, wf_nodes, first_vn)
            mapping[first_vn] = first_sn_idx
            resource_deduction_history.append((first_sn_idx, first_vn))
            first_placed = True
            
            # 记录第一个节点的放置顺序和资源状态
            if return_placement_order:
                first_sn_node = sn_nodes[first_sn_idx]
                placement_order.append((
                    first_vn,
                    first_sn_idx,
                    {
                        'cpu': float(first_sn_node.get('cpu', 0.0)),
                        'memory': float(first_sn_node.get('memory', 0.0)),
                        'disk': float(first_sn_node.get('disk', 0.0))
                    }
                ))
            break
    
    # 如果无法放置第一个节点，返回空映射
    if not first_placed:
        return {}, []
    
    # 8. BFS 扩展放置
    placed_vn: Set[int] = {first_vn}
    queue = [first_vn]
    
    while queue and len(placed_vn) < N1:
        new_placed: List[int] = []
        
        for vi in queue:
            vi_sn_idx = mapping[vi]
            vi_sn_id = int(sn_nodes[vi_sn_idx].get('id', vi_sn_idx))
            
            # 找到 vi 的未放置邻居
            unplaced_neighbors = [u for u in vn_neighbors[vi] if u not in placed_vn]
            
            # 对每个邻居尝试放置
            for u in unplaced_neighbors:
                # 首先尝试放在同一个 SN 节点上
                # 注意：temp_mapping 应该只包含当前轮新放置但尚未扣减资源的节点
                current_round_temp = {vn: sn for vn, sn in mapping.items() if vn in new_placed}
                if _check_sn_resource_for_placement(sn_nodes, vi_sn_idx, wf_nodes, u, temp_mapping=current_round_temp):
                    # 资源足够，立即扣减
                    _deduct_resource(sn_nodes, vi_sn_idx, wf_nodes, u)
                    mapping[u] = vi_sn_idx
                    placed_vn.add(u)
                    new_placed.append(u)
                    resource_deduction_history.append((vi_sn_idx, u))
                    
                    # 记录放置顺序和资源状态
                    if return_placement_order:
                        sn_node = sn_nodes[vi_sn_idx]
                        placement_order.append((
                            u,
                            vi_sn_idx,
                            {
                                'cpu': float(sn_node.get('cpu', 0.0)),
                                'memory': float(sn_node.get('memory', 0.0)),
                                'disk': float(sn_node.get('disk', 0.0))
                            }
                        ))
                    continue
                
                # 否则在 k 跳邻居中找
                k = 1
                # 允许扩展到整个SN网络（最大跳数为SN节点数）
                max_k = len(sn_nodes)
                placed = False
                
                while k <= max_k and not placed:
                    k_hop_neighbors = _get_sn_k_hop_neighbors_from_topo(sn_topo, vi_sn_id, k)
                    # 将 k 跳邻居的 ID 转换为索引
                    k_hop_neighbor_indices = set()
                    for sn_id in k_hop_neighbors:
                        if sn_id in sn_id_to_idx:
                            k_hop_neighbor_indices.add(sn_id_to_idx[sn_id])
                    
                    # 按优先级列表顺序尝试
                    # 同样，temp_mapping 只包含当前轮新放置的节点
                    current_round_temp = {vn: sn for vn, sn in mapping.items() if vn in new_placed}
                    for sn_idx in priority_lists[u]:
                        if sn_idx in k_hop_neighbor_indices and _check_sn_resource_for_placement(sn_nodes, sn_idx, wf_nodes, u, temp_mapping=current_round_temp):
                            # 资源足够，立即扣减
                            _deduct_resource(sn_nodes, sn_idx, wf_nodes, u)
                            mapping[u] = sn_idx
                            placed_vn.add(u)
                            new_placed.append(u)
                            resource_deduction_history.append((sn_idx, u))
                            placed = True
                            
                            # 记录放置顺序和资源状态
                            if return_placement_order:
                                sn_node = sn_nodes[sn_idx]
                                placement_order.append((
                                    u,
                                    sn_idx,
                                    {
                                        'cpu': float(sn_node.get('cpu', 0.0)),
                                        'memory': float(sn_node.get('memory', 0.0)),
                                        'disk': float(sn_node.get('disk', 0.0))
                                    }
                                ))
                            break
                    # 如果当前k跳内没有找到可放置节点，扩展到k+1跳
                    k += 1
                
                # 如果无法放置该节点，回滚所有资源扣减
                if not placed:
                    _rollback_resource_deductions(sn_nodes, wf_nodes, resource_deduction_history)
                    return {}, []
        
        # 更新队列：按度降序；若度相同，则按资源需求降序
        queue = sorted(
            new_placed,
            key=lambda i: (vn_degrees[i], vn_resource_demands[i]),
            reverse=True
        )
    
    # 检查是否所有节点都已放置
    if len(placed_vn) < N1:
        # 部分节点未放置，回滚所有资源扣减
        _rollback_resource_deductions(sn_nodes, wf_nodes, resource_deduction_history)
        return {}, []
    
    if return_placement_order:
        return mapping, placement_order
    else:
        return mapping, []


def generate_pretrain_dataset(
    sn_topo_path: str,
    workflow_topo_path: str,
    workflow_noderank_path: str,
    output_path: str,
    test_output_path: str = None,
    workflows_per_episode: int = 10,
    num_episodes: int = 50,
    test_mode: bool = False,
    test_episode_idx: int = 0
) -> None:
    """生成预训练数据集并保存到文件。
    
    Args:
        sn_topo_path: 底层网络拓扑文件路径
        workflow_topo_path: Workflow 拓扑文件路径
        workflow_noderank_path: Workflow NodeRank 文件路径
        output_path: 输出数据集文件路径（.pt 格式）
        workflows_per_episode: 每个 episode 放置的 workflow 数量
        num_episodes: episode 数量（重复次数）
        test_mode: 是否启用测试模式（打印标签和放置动作）
        test_episode_idx: 测试模式下的 episode 索引（默认第一个 episode）
    """
    print("=" * 60)
    print("开始生成预训练数据集")
    print("=" * 60)
    print(f"底层网络拓扑: {sn_topo_path}")
    print(f"Workflow 拓扑: {workflow_topo_path}")
    print(f"Workflow NodeRank: {workflow_noderank_path}")
    print(f"每个 episode 放置 {workflows_per_episode} 个 workflow")
    print(f"共 {num_episodes} 个 episode")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    # 加载基础数据
    print("\n加载拓扑和 NodeRank 数据...")
    base_sn_topo = _load_json(sn_topo_path)
    base_workflow_topo = _load_json(workflow_topo_path)
    wf_rank_data = _load_json(workflow_noderank_path)
    workflow_noderank: List[float] = list(wf_rank_data['noderank'])
    
    # 计算SN最大容量用于归一化
    sn_max_capacity = _compute_sn_max_capacity(base_sn_topo['nodes'])
    print(f"\n  SN最大容量（用于归一化）:")
    print(f"    CPU: {sn_max_capacity['cpu_max']}")
    print(f"    Memory: {sn_max_capacity['mem_max']}")
    print(f"    Disk: {sn_max_capacity['disk_max']}")
    print(f"    Bandwidth: {sn_max_capacity['bw_max']}")
    print(f"    Comm Bandwidth: {sn_max_capacity['comm_bw_max']}")
    
    print(f"\n  底层网络节点数: {len(base_sn_topo['nodes'])}")
    print(f"  底层网络边数: {len(base_sn_topo.get('links', []))}")
    print(f"  Workflow 节点数: {len(base_workflow_topo['nodes'])}")
    print(f"  Workflow 边数: {len(base_workflow_topo.get('links', []))}")
    print(f"  Workflow NodeRank 维度: {len(workflow_noderank)}")
    
    # 生成一条测试样本（未放置任何任务的 SN + workflow_1）
    if test_output_path:
        print("\n生成单条测试样本（初始 SN 与 workflow）...")
        base_sn_noderank = _compute_sn_noderank(
            base_sn_topo['nodes'], base_sn_topo.get('links', []), directed=bool(base_sn_topo.get('directed', False))
        )
        # 使用归一化
        test_workflow_graph = _topology_to_pyg_data(base_workflow_topo, is_workflow=True, sn_max_capacity=sn_max_capacity)
        test_substrate_graph = _topology_to_pyg_data(
            base_sn_topo,
            is_workflow=False,
            sn_max_capacity=sn_max_capacity
        )
        N1_test = test_workflow_graph.x.size(0)
        N2_test = test_substrate_graph.x.size(0)
        # 直接使用原始noderank，无需重排
        test_y = torch.tensor(
            np.tile(base_sn_noderank.reshape(1, N2_test), (N1_test, 1)),
            dtype=torch.float
        )
        os.makedirs(os.path.dirname(test_output_path) if os.path.dirname(test_output_path) else '.', exist_ok=True)
        torch.save({
            'samples': [{
                'workflow_graph': test_workflow_graph,
                'substrate_graph': test_substrate_graph,
                'label': test_y
            }],
            'info': {
                'type': 'single_test_sample',
                'sn_topo_path': sn_topo_path,
                'workflow_topo_path': workflow_topo_path,
                'workflow_noderank_path': workflow_noderank_path,
                'sn_max_capacity': sn_max_capacity
            }
        }, test_output_path)
        print(f"测试样本已保存到: {test_output_path}")

    # 生成训练样本
    samples: List[Dict] = []
    total_samples = num_episodes * workflows_per_episode
    
    print(f"\n开始生成 {total_samples} 个样本...")
    if test_mode:
        print(f"\n{'='*60}")
        print(f"测试模式：将打印 Episode {test_episode_idx} 的详细放置信息")
        print(f"{'='*60}\n")
    
    with tqdm(total=total_samples, desc="生成样本") as pbar:
        for episode_idx in range(num_episodes):
            # 重置 SN 到初始状态（深拷贝）
            sn_topo = json.loads(json.dumps(base_sn_topo))
            
            # 测试模式：只在指定的 episode 打印详细信息
            is_test_episode = test_mode and (episode_idx == test_episode_idx)
            # 最后一个 episode：打印详细状态和标签
            is_last_episode = (episode_idx == num_episodes - 1)
            
            # 只在最后一个 episode 打印初始状态
            if is_last_episode:
                print(f"\n{'='*80}")
                print(f"最后一个 Episode {episode_idx + 1}/{num_episodes}")
                print(f"{'='*80}")
                
                # 计算初始 SN 的 noderank 和资源状态
                initial_sn_noderank = _compute_sn_noderank(
                    sn_topo['nodes'], 
                    sn_topo.get('links', []), 
                    directed=bool(sn_topo.get('directed', False))
                )
                
                # 打印 SN 初始状态
                print(f"\n【最后一个 Episode {episode_idx + 1} 初始状态】")
                print(f"SN 节点数: {len(sn_topo['nodes'])}")
                print(f"SN 边数: {len(sn_topo.get('links', []))}")
                
                # 计算总资源
                total_cpu = sum(float(n.get('cpu', 0.0)) for n in sn_topo['nodes'])
                total_mem = sum(float(n.get('memory', 0.0)) for n in sn_topo['nodes'])
                total_disk = sum(float(n.get('disk', 0.0)) for n in sn_topo['nodes'])
                print(f"SN 总资源: CPU={total_cpu:.2f}, Memory={total_mem:.2f}, Disk={total_disk:.2f}")
                
                # 打印前5个节点的 NodeRank 和资源
                print(f"\nSN NodeRank 前5名（初始状态）:")
                sn_noderank_sorted = sorted(enumerate(initial_sn_noderank), key=lambda x: x[1], reverse=True)
                for rank, (sn_idx, rank_val) in enumerate(sn_noderank_sorted[:5], 1):
                    sn_node = sn_topo['nodes'][sn_idx]
                    sn_id = sn_node.get('id', sn_idx)
                    cpu = sn_node.get('cpu', 0.0)
                    mem = sn_node.get('memory', 0.0)
                    disk = sn_node.get('disk', 0.0)
                    print(f"  {rank}. SN节点 {sn_idx} (ID={sn_id}): NodeRank={rank_val:.6f}, "
                          f"资源 [CPU={cpu:.2f}, Mem={mem:.2f}, Disk={disk:.2f}]")
            else:
                # 非最后一个episode也需要计算总资源（用于最后的总结）
                total_cpu = sum(float(n.get('cpu', 0.0)) for n in sn_topo['nodes'])
                total_mem = sum(float(n.get('memory', 0.0)) for n in sn_topo['nodes'])
                total_disk = sum(float(n.get('disk', 0.0)) for n in sn_topo['nodes'])
            
            for wf_idx in range(workflows_per_episode):
                # 计算当前 SN 的 noderank
                sn_noderank = _compute_sn_noderank(
                    sn_topo['nodes'], 
                    sn_topo.get('links', []), 
                    directed=bool(sn_topo.get('directed', False))
                )
                
                # x: 当前的 workflow 与 SN 图（放置前状态，使用归一化）
                workflow_graph = _topology_to_pyg_data(base_workflow_topo, is_workflow=True, sn_max_capacity=sn_max_capacity)
                substrate_graph = _topology_to_pyg_data(
                    sn_topo,
                    is_workflow=False,
                    sn_max_capacity=sn_max_capacity
                )
                
                # y: 将 SN 的 noderank 重复 N1 行（直接使用原始noderank，无需重排）
                N1 = workflow_graph.x.size(0)
                N2 = substrate_graph.x.size(0)
                y = torch.tensor(
                    np.tile(sn_noderank.reshape(1, N2), (N1, 1)), 
                    dtype=torch.float
                )
                
                # 计算当前 SN 资源状态（所有episode都需要，但只在最后一个episode打印）
                current_total_cpu = sum(float(n.get('cpu', 0.0)) for n in sn_topo['nodes'])
                current_total_mem = sum(float(n.get('memory', 0.0)) for n in sn_topo['nodes'])
                current_total_disk = sum(float(n.get('disk', 0.0)) for n in sn_topo['nodes'])
                
                # 只在最后一个 episode 打印 workflow 的标签信息
                if is_last_episode:
                    print(f"\n  --- Workflow {wf_idx + 1}/{workflows_per_episode} ---")
                    print(f"  VN 节点数: {N1}, SN 节点数: {N2}")
                    print(f"  标签 y 形状: {y.shape} (VN节点数 × SN节点数)")
                    print(f"  当前 SN 剩余资源: CPU={current_total_cpu:.2f}, Memory={current_total_mem:.2f}, Disk={current_total_disk:.2f}")
                    
                    # 打印标签矩阵的统计信息
                    y_np = y.numpy()
                    print(f"  标签矩阵统计:")
                    print(f"    最小值: {y_np.min():.6f}, 最大值: {y_np.max():.6f}, 平均值: {y_np.mean():.6f}")
                    print(f"    每行和: {[f'{row_sum:.6f}' for row_sum in y_np.sum(axis=1)]}")
                    
                    # 打印每个 VN 节点对应的最大 NodeRank 值（即最优先的 SN 节点）
                    print(f"  每个 VN 节点的最大 NodeRank 值（最优先 SN 节点）:")
                    for vn_idx in range(N1):
                        max_rank_idx = np.argmax(y_np[vn_idx])
                        max_rank_val = y_np[vn_idx, max_rank_idx]
                        sn_node = sn_topo['nodes'][max_rank_idx]
                        sn_id = sn_node.get('id', max_rank_idx)
                        sn_cpu = sn_node.get('cpu', 0.0)
                        sn_mem = sn_node.get('memory', 0.0)
                        sn_disk = sn_node.get('disk', 0.0)
                        print(f"    VN节点 {vn_idx} -> SN节点 {max_rank_idx} (ID={sn_id}): "
                              f"NodeRank={max_rank_val:.6f}, 剩余资源 [CPU={sn_cpu:.2f}, Mem={sn_mem:.2f}, Disk={sn_disk:.2f}]")
                    
                    # 打印完整的标签矩阵和所有SN节点状态
                    print(f"\n  【最后一个Episode - 详细标签矩阵】")
                    print(f"  完整标签矩阵 y (形状: {y.shape}):")
                    for vn_idx in range(N1):
                        print(f"    VN节点 {vn_idx}: {y_np[vn_idx]}")
                    
                    print(f"\n  所有SN节点的NodeRank和资源状态（降序）:")
                    sn_noderank_sorted = sorted(enumerate(sn_noderank), key=lambda x: x[1], reverse=True)
                    for rank, (sn_idx, rank_val) in enumerate(sn_noderank_sorted, 1):
                        sn_node = sn_topo['nodes'][sn_idx]
                        sn_id = sn_node.get('id', sn_idx)
                        cpu = sn_node.get('cpu', 0.0)
                        mem = sn_node.get('memory', 0.0)
                        disk = sn_node.get('disk', 0.0)
                        print(f"    {rank}. SN节点 {sn_idx} (ID={sn_id}): NodeRank={rank_val:.6f}, "
                              f"剩余资源 [CPU={cpu:.2f}, Mem={mem:.2f}, Disk={disk:.2f}]")
                else:
                    # 非最后一个episode也需要定义y_np（用于后续可能的操作）
                    y_np = y.numpy()
                
                # 测试模式：打印详细标签 y
                if is_test_episode:
                    print(f"\n{'='*60}")
                    print(f"Episode {episode_idx + 1}, Workflow {wf_idx + 1}")
                    print(f"{'='*60}")
                    print(f"\n【放置前标签 y (NodeRank 矩阵)】")
                    print(f"形状: {y.shape} (VN节点数 × SN节点数)")
                    print(f"标签矩阵:")
                    y_np = y.numpy()
                    for vn_idx in range(N1):
                        print(f"  VN节点 {vn_idx}: {y_np[vn_idx]}")
                    print(f"\nSN NodeRank 值 (降序):")
                    sn_noderank_sorted = sorted(enumerate(sn_noderank), key=lambda x: x[1], reverse=True)
                    for sn_idx, rank_val in sn_noderank_sorted[:5]:  # 只显示前5个
                        sn_node = sn_topo['nodes'][sn_idx]
                        sn_id = sn_node.get('id', sn_idx)
                        cpu = sn_node.get('cpu', 0.0)
                        mem = sn_node.get('memory', 0.0)
                        disk = sn_node.get('disk', 0.0)
                        print(f"  SN节点 {sn_idx} (ID={sn_id}): NodeRank={rank_val:.6f}, "
                              f"剩余资源 [CPU={cpu:.2f}, Mem={mem:.2f}, Disk={disk:.2f}]")
                
                # 保存样本
                samples.append({
                    'workflow_graph': workflow_graph,
                    'substrate_graph': substrate_graph,
                    'label': y
                })
                
                pbar.update(1)
                
                # 放置该 workflow 并更新 SN 资源（使用 BFS 策略）
                if is_test_episode or is_last_episode:
                    mapping, placement_order = _greedy_place_workflow(
                        base_workflow_topo,
                        sn_topo,
                        sn_topo['nodes'],
                        sn_noderank,
                        workflow_noderank,
                        k_hop=1,
                        return_placement_order=True
                    )
                else:
                    mapping, _ = _greedy_place_workflow(
                        base_workflow_topo,
                        sn_topo,
                        sn_topo['nodes'],
                        sn_noderank,
                        workflow_noderank,
                        k_hop=1,
                        return_placement_order=False
                    )
                    placement_order = []
                
                # 测试模式或最后一个episode：打印放置动作（按放置顺序）
                if (is_test_episode or is_last_episode) and y_np is not None:
                    if is_last_episode:
                        print(f"\n  【最后一个Episode - 详细放置动作】（按放置顺序）")
                    else:
                        print(f"\n【放置动作】（按放置顺序）")
                    print(f"  成功放置 {len(mapping)}/{N1} 个 VN 节点")
                    wf_nodes = base_workflow_topo['nodes']
                    
                    # 按放置顺序打印
                    if placement_order:
                        for place_idx, (vn_idx, sn_idx, sn_resources_after) in enumerate(placement_order, 1):
                            vn_node = wf_nodes[vn_idx]
                            sn_node = sn_topo['nodes'][sn_idx]
                            vn_id = vn_node.get('id', vn_idx)
                            sn_id = sn_node.get('id', sn_idx)
                            vn_cpu = vn_node.get('cpu', 0.0)
                            vn_mem = vn_node.get('memory', 0.0)
                            vn_disk = vn_node.get('disk', 0.0)
                            
                            print(f"\n    步骤 {place_idx}: 放置 VN节点 {vn_idx} (ID={vn_id})")
                            print(f"      VN节点需求: CPU={vn_cpu:.2f}, Mem={vn_mem:.2f}, Disk={vn_disk:.2f}")
                            print(f"      -> SN节点 {sn_idx} (ID={sn_id})")
                            print(f"      放置前 SN 节点资源: CPU={sn_resources_after['cpu'] + vn_cpu:.2f}, "
                                  f"Mem={sn_resources_after['memory'] + vn_mem:.2f}, "
                                  f"Disk={sn_resources_after['disk'] + vn_disk:.2f}")
                            print(f"      放置后 SN 节点剩余资源: CPU={sn_resources_after['cpu']:.2f}, "
                                  f"Mem={sn_resources_after['memory']:.2f}, Disk={sn_resources_after['disk']:.2f}")
                            print(f"      标签 y[{vn_idx}, {sn_idx}] = {y_np[vn_idx, sn_idx]:.6f}")
                            print(f"      该SN节点在所有SN节点中的NodeRank排名: "
                                  f"{sorted(enumerate(sn_noderank), key=lambda x: x[1], reverse=True).index((sn_idx, sn_noderank[sn_idx])) + 1}/{N2}")
                    else:
                        # 如果无法获取放置顺序，使用原来的方式
                        print(f"  放置映射 (VN节点索引 -> SN节点索引):")
                        for vn_idx in sorted(mapping.keys()):
                            sn_idx = mapping[vn_idx]
                            vn_node = wf_nodes[vn_idx]
                            sn_node = sn_topo['nodes'][sn_idx]
                            vn_id = vn_node.get('id', vn_idx)
                            sn_id = sn_node.get('id', sn_idx)
                            vn_cpu = vn_node.get('cpu', 0.0)
                            vn_mem = vn_node.get('memory', 0.0)
                            vn_disk = vn_node.get('disk', 0.0)
                            sn_cpu = sn_node.get('cpu', 0.0)
                            sn_mem = sn_node.get('memory', 0.0)
                            sn_disk = sn_node.get('disk', 0.0)
                            print(f"    VN节点 {vn_idx} (ID={vn_id}) [需求: CPU={vn_cpu:.2f}, Mem={vn_mem:.2f}, Disk={vn_disk:.2f}]")
                            print(f"      -> SN节点 {sn_idx} (ID={sn_id}) [剩余: CPU={sn_cpu:.2f}, Mem={sn_mem:.2f}, Disk={sn_disk:.2f}]")
                            print(f"      -> 标签 y[{vn_idx}, {sn_idx}] = {y_np[vn_idx, sn_idx]:.6f}")
                            print(f"      -> 该SN节点在所有SN节点中的NodeRank排名: "
                                  f"{sorted(enumerate(sn_noderank), key=lambda x: x[1], reverse=True).index((sn_idx, sn_noderank[sn_idx])) + 1}/{N2}")
                    
                    # 检查未放置的节点
                    unplaced = [i for i in range(N1) if i not in mapping]
                    if unplaced:
                        print(f"\n  未成功放置的 VN 节点: {unplaced}")
                        print(f"  未放置节点的资源需求:")
                        for vn_idx in unplaced:
                            vn_node = wf_nodes[vn_idx]
                            vn_id = vn_node.get('id', vn_idx)
                            vn_cpu = vn_node.get('cpu', 0.0)
                            vn_mem = vn_node.get('memory', 0.0)
                            vn_disk = vn_node.get('disk', 0.0)
                            print(f"    VN节点 {vn_idx} (ID={vn_id}): CPU={vn_cpu:.2f}, Mem={vn_mem:.2f}, Disk={vn_disk:.2f}")
                    print(f"\n")
                
                # 只在最后一个 episode 打印放置后的状态更新
                if is_last_episode:
                    if len(mapping) > 0:
                        print(f"  ✓ Workflow {wf_idx + 1} 放置完成: {len(mapping)}/{N1} 个节点成功放置")
                        # 计算放置后的 SN 资源状态
                        after_total_cpu = sum(float(n.get('cpu', 0.0)) for n in sn_topo['nodes'])
                        after_total_mem = sum(float(n.get('memory', 0.0)) for n in sn_topo['nodes'])
                        after_total_disk = sum(float(n.get('disk', 0.0)) for n in sn_topo['nodes'])
                        print(f"  放置后 SN 剩余资源: CPU={after_total_cpu:.2f}, Memory={after_total_mem:.2f}, Disk={after_total_disk:.2f}")
                        print(f"  资源消耗: CPU={current_total_cpu - after_total_cpu:.2f}, "
                              f"Memory={current_total_mem - after_total_mem:.2f}, "
                              f"Disk={current_total_disk - after_total_disk:.2f}")
                    else:
                        print(f"  ✗ Workflow {wf_idx + 1} 放置失败: 无法放置任何节点")
            
            # 只在最后一个 episode 打印结束总结
            if is_last_episode:
                final_total_cpu = sum(float(n.get('cpu', 0.0)) for n in sn_topo['nodes'])
                final_total_mem = sum(float(n.get('memory', 0.0)) for n in sn_topo['nodes'])
                final_total_disk = sum(float(n.get('disk', 0.0)) for n in sn_topo['nodes'])
                print(f"\n【最后一个 Episode {episode_idx + 1} 结束总结】")
                print(f"  初始资源: CPU={total_cpu:.2f}, Memory={total_mem:.2f}, Disk={total_disk:.2f}")
                print(f"  最终剩余资源: CPU={final_total_cpu:.2f}, Memory={final_total_mem:.2f}, Disk={final_total_disk:.2f}")
                print(f"  总资源消耗: CPU={total_cpu - final_total_cpu:.2f}, "
                      f"Memory={total_mem - final_total_mem:.2f}, "
                      f"Disk={total_disk - final_total_disk:.2f}")
                print(f"  资源利用率: CPU={(total_cpu - final_total_cpu) / total_cpu * 100:.2f}%, "
                      f"Memory={(total_mem - final_total_mem) / total_mem * 100:.2f}%, "
                      f"Disk={(total_disk - final_total_disk) / total_disk * 100:.2f}%")
                print(f"{'='*80}\n")
    
    # 保存数据集
    print(f"\n保存数据集到 {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    dataset_info = {
        'num_samples': len(samples),
        'workflows_per_episode': workflows_per_episode,
        'num_episodes': num_episodes,
        'sn_topo_path': sn_topo_path,
        'workflow_topo_path': workflow_topo_path,
        'workflow_noderank_path': workflow_noderank_path,
        'sn_max_capacity': sn_max_capacity,  # 保存归一化参数
        'normalized': True,  # 标记数据已归一化
    }
    
    torch.save({
        'samples': samples,
        'info': dataset_info
    }, output_path)
    
    print("=" * 60)
    print(f"数据集生成完成！")
    print(f"  总样本数: {len(samples)}")
    print(f"  文件大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成预训练数据集')
    parser.add_argument('--sn_topo', type=str, 
                       default='/home/yc2/mrt/a/topo/SN_topology.json',
                       help='底层网络拓扑文件路径')
    parser.add_argument('--workflow_topo', type=str,
                       default='/home/yc2/mrt/a/workflow_topo/workflow1_topo.json',
                       help='Workflow 拓扑文件路径')
    parser.add_argument('--workflow_noderank', type=str,
                       default='/home/yc2/mrt/a/workflow_topo/workflow1_noderank.json',
                       help='Workflow NodeRank 文件路径')
    parser.add_argument('--output', type=str,
                       default='/home/yc2/mrt/a/pretrain_data/pretrain_dataset.pt',
                       help='输出数据集文件路径')
    parser.add_argument('--test_output', type=str,
                       default='/home/yc2/mrt/a/pretrain_data/test_sample.pt',
                       help='测试样本输出文件路径（单条）')
    parser.add_argument('--workflows_per_episode', type=int, default=5,
                       help='每个 episode 放置的 workflow 数量')
    parser.add_argument('--num_episodes', type=int, default=400,
                       help='Episode 数量（重复次数）')
    parser.add_argument('--test_mode', action='store_true',
                       help='启用测试模式，打印放置前标签和放置动作')
    parser.add_argument('--test_episode_idx', type=int, default=0,
                       help='测试模式下要打印的 episode 索引（默认 0）')
    
    args = parser.parse_args()
    
    generate_pretrain_dataset(
        sn_topo_path=args.sn_topo,
        workflow_topo_path=args.workflow_topo,
        workflow_noderank_path=args.workflow_noderank,
        output_path=args.output,
        test_output_path=args.test_output,
        workflows_per_episode=args.workflows_per_episode,
        num_episodes=args.num_episodes,
        test_mode=args.test_mode,
        test_episode_idx=args.test_episode_idx
    )


if __name__ == '__main__':
    main()
