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

- 这个是使用最开始的偏置生成时的计算，即输入偏置
"""

import torch
import os
import json
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from torch_geometric.data import Data
from tqdm import tqdm
from topo.calculate_noderank_2 import build_adjacency_info, calculate_forward_probability
from topo.SN_topo_addbias import add_bias_to_topology


def _load_json(path: str) -> Dict:
    """
    加载 JSON 文件
    
    Args:
        path: JSON 文件路径（支持绝对路径和相对路径）
    
    Returns:
        解析后的字典
    
    Raises:
        FileNotFoundError: 文件不存在时抛出，包含详细错误信息
        json.JSONDecodeError: JSON 解析错误时抛出
    """
    # 转换为绝对路径
    if not os.path.isabs(path):
        # 相对路径：相对于脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(script_dir, path)
    else:
        abs_path = path
    
    # 检查文件是否存在
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"文件不存在: {abs_path}\n"
            f"原始路径: {path}\n"
            f"请检查路径是否正确，或使用 --sn_topo/--workflow_topo 等参数指定正确的路径"
        )
    
    # 检查是否为文件
    if not os.path.isfile(abs_path):
        raise ValueError(f"路径不是文件: {abs_path}")
    
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"JSON 解析错误: {abs_path}\n"
            f"错误详情: {str(e)}",
            e.doc,
            e.pos
        )


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


def _extract_constraint_node(workflow_topo: Dict) -> int:
    """从 workflow 拓扑中提取唯一的 constraint_node。"""
    constraint_nodes = {
        int(node['constraint_node'])
        for node in workflow_topo.get('nodes', [])
        if node.get('constraint_node') is not None
    }
    if not constraint_nodes:
        raise ValueError("workflow 拓扑中未找到 constraint_node 字段")
    if len(constraint_nodes) > 1:
        raise ValueError(f"期望唯一的 constraint_node，但发现多个: {sorted(constraint_nodes)}")
    return constraint_nodes.pop()


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


def _generate_bias_for_constraint_nodes(
    sn_nodes: List[Dict],
    sn_max_capacity: Dict[str, float],
    constraint_sn_node_ids: List[int],
    bias_factor: float = 0.5
) -> List[Dict]:
    """
    为约束节点生成偏置资源（仅用于计算noderank，不影响实际资源）
    
    Args:
        sn_nodes: SN节点列表
        sn_max_capacity: SN最大容量字典，用于归一化
        constraint_sn_node_ids: 需要添加偏置的SN节点ID列表
        bias_factor: 偏置因子，默认为0.5（归一化资源的0.5倍）
    
    Returns:
        修改后的SN节点列表（深拷贝，仅用于计算noderank）
    """
    # 深拷贝节点列表
    biased_nodes = json.loads(json.dumps(sn_nodes))
    
    # 创建SN节点ID到索引的映射
    sn_id_to_idx = {}
    for idx, node in enumerate(sn_nodes):
        sn_id = int(node.get('id', idx))
        sn_id_to_idx[sn_id] = idx
    
    # 为每个约束节点添加偏置
    for sn_id in constraint_sn_node_ids:
        if sn_id in sn_id_to_idx:
            sn_idx = sn_id_to_idx[sn_id]
            biased_node = biased_nodes[sn_idx]
            
            # 计算归一化的偏置值（基于最大容量）
            cpu_bias = bias_factor * sn_max_capacity['cpu_max']
            mem_bias = bias_factor * sn_max_capacity['mem_max']
            disk_bias = bias_factor * sn_max_capacity['disk_max']
            comm_bw_bias = bias_factor * sn_max_capacity['comm_bw_max']
            
            # 添加偏置（仅用于计算noderank）
            biased_node['cpu'] = float(biased_node.get('cpu', 0.0)) + cpu_bias
            biased_node['memory'] = float(biased_node.get('memory', 0.0)) + mem_bias
            biased_node['disk'] = float(biased_node.get('disk', 0.0)) + disk_bias
            biased_node['comm_bandwidth'] = float(biased_node.get('comm_bandwidth', biased_node.get('bandwidth', 0.0))) + comm_bw_bias
    
    return biased_nodes


def _compute_sn_noderank(
    nodes: List[Dict],
    links: List[Dict],
    directed: bool = False,
    num_iterations: int = 2
) -> np.ndarray:
    """
    计算底层网络节点的 NodeRank。
    资源评估 H(u) = (cpu + bias_cpu) * bandwidth
    然后走 calculate_noderank_2 的传播与归一化逻辑。
    
    Args:
        nodes: SN节点列表
        links: SN边列表
        directed: 是否为有向图
        num_iterations: NodeRank迭代次数（默认2）
    """
    n = len(nodes)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    # 构建出链信息（用于计算资源评估）
    outgoing_links = {i: [] for i in range(n)}
    for link in links:
        source = int(link['source'])
        target = int(link['target'])
        bw = float(link.get('bandwidth', 1.0))
        if source < n and target < n:
            outgoing_links[source].append({'target': target, 'bandwidth': bw})
            if not directed:
                outgoing_links[target].append({'target': source, 'bandwidth': bw})

    # 步骤1：计算初始资源评估 H(u) = (cpu + bias_cpu) * bandwidth
    H = np.zeros(n, dtype=np.float64)
    for i, node in enumerate(nodes):
        cpu = float(node.get('cpu', 0.0))
        bias_cpu = float(node.get('bias_cpu', 0.0))
        bandwidth = float(node.get('bandwidth', node.get('comm_bandwidth', 0.0)))
        H[i] = (cpu + bias_cpu) * bandwidth

    # 步骤2：计算初始 NodeRank
    H_sum = np.sum(H) if np.sum(H) > 0 else 1.0
    NR_0 = H / H_sum

    # 步骤3：构建邻接信息（调用 calculate_noderank_2.py 的实现）
    adj_info = build_adjacency_info(nodes, links, directed=directed)
    adjacency = adj_info['adjacency']

    # 步骤4：计算前向概率矩阵（调用 calculate_noderank_2.py 的实现）
    pF = calculate_forward_probability(H, adjacency)

    # 步骤5：迭代计算 NodeRank（使用 calculate_noderank_2.py 的逻辑）
    PJ_u = 0.80
    PF_u = 0.20
    num_first_transforms = num_iterations

    NR_curr = NR_0.copy()
    for _ in range(num_first_transforms):
        NR_next = NR_curr + PF_u * (pF @ NR_curr)
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
    return_placement_order: bool = False,
    failure_log: Optional[List[str]] = None
) -> Tuple[Dict[int, int], List[Tuple[int, int, Dict[str, float]]]]:
    """
    使用 BFS 扩展策略将一个 workflow 的所有节点贪心放置到底层网络。
    参考 test.py 的 place_with_bfs_strategy 逻辑。
    
    放置规则：
    1. 首先处理有 constraint_node 的 VN 节点，直接放置到指定的 SN 节点
    2. 选择资源占用最大的 VN 节点作为第一个（排除已放置的约束节点）
    3. 基于 VN 图的 BFS 扩展，优先放在同一 SN 节点上
    4. 如果资源不足，在 k 跳邻居中查找
    5. 按 NodeRank 排序确定优先级
    6. constraint_node 不为空的节点始终不入队列
    
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
    
    # 获取 SN 节点 ID 到索引的映射
    sn_id_to_idx = {}
    for idx, node in enumerate(sn_nodes):
        sn_id = int(node.get('id', idx))
        sn_id_to_idx[sn_id] = idx
    
    # 初始化映射和记录
    mapping: Dict[int, int] = {}  # VN节点索引 -> SN节点索引
    placement_order: List[Tuple[int, int, Dict[str, float]]] = []  # 放置顺序记录
    resource_deduction_history: List[Tuple[int, int]] = []  # 资源扣减历史记录
    constraint_placed_vn: Set[int] = set()  # 已放置的约束节点集合
    
    # ========== 步骤1：先处理有 constraint_node 的节点 ==========
    for vn_idx, vn_node in enumerate(wf_nodes):
        constraint_node_id = vn_node.get('constraint_node')
        if constraint_node_id is not None:
            # 找到对应的 SN 节点索引
            if constraint_node_id in sn_id_to_idx:
                sn_idx = sn_id_to_idx[constraint_node_id]
                
                # 检查资源是否足够
                if _check_sn_resource_for_placement(sn_nodes, sn_idx, wf_nodes, vn_idx, temp_mapping=None):
                    # 资源足够，立即扣减并放置
                    _deduct_resource(sn_nodes, sn_idx, wf_nodes, vn_idx)
                    mapping[vn_idx] = sn_idx
                    constraint_placed_vn.add(vn_idx)
                    resource_deduction_history.append((sn_idx, vn_idx))
                    
                    # 记录放置顺序和资源状态
                    if return_placement_order:
                        sn_node = sn_nodes[sn_idx]
                        placement_order.append((
                            vn_idx,
                            sn_idx,
                            {
                                'cpu': float(sn_node.get('cpu', 0.0)),
                                'memory': float(sn_node.get('memory', 0.0)),
                                'disk': float(sn_node.get('disk', 0.0))
                            }
                        ))
                else:
                    # 资源不足，无法放置约束节点，返回空映射
                    if failure_log is not None:
                        failure_log.append(
                            f"约束节点放置失败: VN {vn_idx} -> SN_ID {constraint_node_id} 资源不足"
                        )
                    _rollback_resource_deductions(sn_nodes, wf_nodes, resource_deduction_history)
                    return {}, []
    
    # 如果所有节点都是约束节点且都已放置，直接返回
    if len(constraint_placed_vn) == N1:
        return mapping, placement_order
    
    # ========== 步骤2：处理非约束节点 ==========
    
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
    
    # 5. 选择第一个非约束 VN 节点（资源占用最大）
    non_constraint_vn = [i for i in range(N1) if i not in constraint_placed_vn]
    if not non_constraint_vn:
        return mapping, placement_order
    
    first_vn = max(non_constraint_vn, key=lambda i: vn_resource_demands[i])
    
    # 6. 放置第一个非约束 VN 节点（遍历优先级列表，找到第一个能放置的）
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
    
    # 如果无法放置第一个节点，回滚约束节点的资源扣减并返回空映射
    if not first_placed:
        if failure_log is not None:
            failure_log.append(f"首个非约束节点放置失败: VN {first_vn} 无可用 SN")
        _rollback_resource_deductions(sn_nodes, wf_nodes, resource_deduction_history)
        return {}, []
    
    # 7. BFS 扩展放置（约束节点不入队列）
    placed_vn: Set[int] = constraint_placed_vn.copy()  # 包含已放置的约束节点
    placed_vn.add(first_vn)
    queue = [first_vn]  # 队列中不包含约束节点
    
    while queue and len(placed_vn) < N1:
        new_placed: List[int] = []
        
        for vi in queue:
            vi_sn_idx = mapping[vi]
            vi_sn_id = int(sn_nodes[vi_sn_idx].get('id', vi_sn_idx))
            
            # 找到 vi 的未放置邻居（排除约束节点）
            unplaced_neighbors = [u for u in vn_neighbors[vi] if u not in placed_vn and u not in constraint_placed_vn]
            
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
                    if failure_log is not None:
                        failure_log.append(
                            f"BFS 放置失败: VN {u} 起点 SN_ID {vi_sn_id}，在 {k-1} 跳内无可用 SN"
                        )
                    _rollback_resource_deductions(sn_nodes, wf_nodes, resource_deduction_history)
                    return {}, []
        
        # 更新队列：按度降序；若度相同，则按资源需求降序（约束节点不入队列）
        queue = sorted(
            [i for i in new_placed if i not in constraint_placed_vn],
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
    test_episode_idx: int = 0,
    bias: float = 0.4,
    noderank_iterations: int = 2
) -> None:
    """生成预训练数据集并保存到文件。"""
    print("=" * 48)
    print("开始生成预训练数据集")
    print(f"SN 输出文件: {sn_topo_path}")
    print(f"workflow: {workflow_topo_path}")
    print(f"bias: {bias}")
    print(f"{workflows_per_episode} workflows / episode × {num_episodes} episodes")
    print("=" * 48)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_sn_input = os.path.join(script_dir, 'topo', 'SN_topology.json')
    biased_sn_path = sn_topo_path

    # 校验输入
    print("\n校验输入文件...")
    for desc, path in {
        '原始 SN 拓扑': base_sn_input,
        'Workflow 拓扑': workflow_topo_path,
        'Workflow NodeRank': workflow_noderank_path
    }.items():
        abs_path = path if os.path.isabs(path) else os.path.join(script_dir, path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{desc} 不存在: {abs_path}")
        print(f"✓ {desc}: {abs_path}")

    # 读取 workflow 并提取唯一 constraint_node
    base_workflow_topo = _load_json(workflow_topo_path)
    constraint_node = _extract_constraint_node(base_workflow_topo)
    print(f"\nconstraint_node: {constraint_node}")

    # 生成带偏置的 SN 拓扑
    print("生成带偏置的 SN_topology_2.json ...")
    add_bias_to_topology(
        input_path=base_sn_input,
        output_path=biased_sn_path,
        bias=bias,
        constraint_node=constraint_node
    )

    # 加载偏置后的 SN 与 workflow noderank
    base_sn_topo = _load_json(biased_sn_path)
    wf_rank_data = _load_json(workflow_noderank_path)
    workflow_noderank: List[float] = list(wf_rank_data['noderank'])

    sn_max_capacity = _compute_sn_max_capacity(base_sn_topo['nodes'])

    # 生成单条测试样本（可选）
    if test_output_path:
        print("\n生成单条测试样本...")
        base_sn_noderank = _compute_sn_noderank(
            base_sn_topo['nodes'],
            base_sn_topo.get('links', []),
            directed=bool(base_sn_topo.get('directed', False)),
            num_iterations=noderank_iterations
        )
        test_workflow_graph = _topology_to_pyg_data(base_workflow_topo, is_workflow=True, sn_max_capacity=sn_max_capacity)
        test_substrate_graph = _topology_to_pyg_data(
            base_sn_topo,
            is_workflow=False,
            sn_max_capacity=sn_max_capacity
        )
        N1_test = test_workflow_graph.x.size(0)
        N2_test = test_substrate_graph.x.size(0)
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
                'sn_topo_path': biased_sn_path,
                'workflow_topo_path': workflow_topo_path,
                'workflow_noderank_path': workflow_noderank_path,
                'sn_max_capacity': sn_max_capacity,
                'bias': bias
            }
        }, test_output_path)
        print(f"测试样本已保存到: {test_output_path}")

    samples: List[Dict] = []
    total_samples = num_episodes * workflows_per_episode
    print(f"\n开始生成 {total_samples} 个样本...")

    with tqdm(total=total_samples, desc="生成样本") as pbar:
        for episode_idx in range(num_episodes):
            sn_topo = json.loads(json.dumps(base_sn_topo))
            is_test_episode = test_mode and (episode_idx == test_episode_idx)
            is_last_episode = (episode_idx == num_episodes - 1)

            if is_last_episode:
                print(f"\n=== 最后一个 Episode {episode_idx + 1}/{num_episodes} ===")

            for wf_idx in range(workflows_per_episode):
                sn_noderank = _compute_sn_noderank(
                    sn_topo['nodes'],
                    sn_topo.get('links', []),
                    directed=bool(sn_topo.get('directed', False)),
                    num_iterations=noderank_iterations
                )

                # 可选打印：放置前每个 SN 节点的 NodeRank
                if is_test_episode or is_last_episode:
                    print(f"\nEpisode {episode_idx + 1}, Workflow {wf_idx + 1} 放置前 NodeRank：")
                    for idx, val in enumerate(sn_noderank):
                        sn_node = sn_topo['nodes'][idx]
                        sn_id = sn_node.get('id', idx)
                        print(f"  SN节点 {idx} (ID={sn_id}): {val:.6f}")

                workflow_graph = _topology_to_pyg_data(base_workflow_topo, is_workflow=True, sn_max_capacity=sn_max_capacity)
                substrate_graph = _topology_to_pyg_data(
                    sn_topo,
                    is_workflow=False,
                    sn_max_capacity=sn_max_capacity
                )

                N1 = workflow_graph.x.size(0)
                N2 = substrate_graph.x.size(0)
                y = torch.tensor(
                    np.tile(sn_noderank.reshape(1, N2), (N1, 1)),
                    dtype=torch.float
                )

                samples.append({
                    'workflow_graph': workflow_graph,
                    'substrate_graph': substrate_graph,
                    'label': y
                })

                pbar.update(1)

                # 放置并记录
                failure_log: List[str] = []
                if is_test_episode or is_last_episode:
                    mapping, placement_order = _greedy_place_workflow(
                        base_workflow_topo,
                        sn_topo,
                        sn_topo['nodes'],
                        sn_noderank,
                        workflow_noderank,
                        k_hop=1,
                        return_placement_order=True,
                        failure_log=failure_log
                    )
                else:
                    mapping, _ = _greedy_place_workflow(
                        base_workflow_topo,
                        sn_topo,
                        sn_topo['nodes'],
                        sn_noderank,
                        workflow_noderank,
                        k_hop=1,
                        return_placement_order=False,
                        failure_log=failure_log
                    )
                    placement_order = []

                if (is_test_episode or is_last_episode):
                    print(f"\nEpisode {episode_idx + 1}, Workflow {wf_idx + 1}: 放置 {len(mapping)}/{N1}")
                    wf_nodes = base_workflow_topo['nodes']
                    if placement_order:
                        for step_idx, (vn_idx, sn_idx, after_res) in enumerate(placement_order, 1):
                            vn_node = wf_nodes[vn_idx]
                            sn_node = sn_topo['nodes'][sn_idx]
                            print(f"  步骤{step_idx}: VN {vn_idx} -> SN {sn_idx} "
                                  f"(前: cpu={after_res['cpu'] + vn_node.get('cpu',0.0):.2f}, "
                                  f"后: cpu={after_res['cpu']:.2f})")
                    else:
                        print("  未获取放置顺序")
                    if len(mapping) < N1:
                        print("  放置失败详情:")
                        if failure_log:
                            for msg in failure_log:
                                print(f"    - {msg}")
                        else:
                            print("    - 未返回详细失败原因")

            if is_last_episode:
                print(f"=== Episode {episode_idx + 1} 完成 ===\n")

    # 保存数据集
    print(f"保存数据集到 {output_path} ...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    dataset_info = {
        'num_samples': len(samples),
        'workflows_per_episode': workflows_per_episode,
        'num_episodes': num_episodes,
        'sn_topo_path': biased_sn_path,
        'workflow_topo_path': workflow_topo_path,
        'workflow_noderank_path': workflow_noderank_path,
        'sn_max_capacity': sn_max_capacity,
        'normalized': True,
        'bias': bias
    }

    torch.save({
        'samples': samples,
        'info': dataset_info
    }, output_path)

    print("=" * 48)
    print(f"完成，样本数: {len(samples)}，文件大小: {os.path.getsize(output_path)/(1024*1024):.2f} MB")
    print("=" * 48)


def main():
    """主函数"""
    import argparse
    
    # 获取脚本所在目录，用于构建相对路径的默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='生成预训练数据集')
    parser.add_argument('--sn_topo', type=str, 
                       default='/home/zrz/AgentVNE/AgentVNE/topo/SN_topology_2.json',
                       help='带偏置的 SN 拓扑输出路径（固定 SN_topology_2.json）')
    parser.add_argument('--workflow_topo', type=str,
                       default=os.path.join(script_dir, 'workflow_topo', 'workflow1_topo.json'),
                       help='Workflow 拓扑文件路径（支持绝对路径和相对路径）')
    parser.add_argument('--workflow_noderank', type=str,
                       default=os.path.join(script_dir, 'workflow_topo', 'workflow1_noderank.json'),
                       help='Workflow NodeRank 文件路径（支持绝对路径和相对路径）')
    parser.add_argument('--output', type=str,
                       default='/home/zrz/AgentVNE/AgentVNE/pretrain_data/pretrain_dataset.pt',
                       help='输出数据集文件路径（支持绝对路径和相对路径）')
    parser.add_argument('--test_output', type=str,
                       default='/home/zrz/AgentVNE/AgentVNE/pretrain_data/test_sample.pt',
                       help='测试样本输出文件路径（单条，支持绝对路径和相对路径）')
    parser.add_argument('--workflows_per_episode', type=int, default=7,
                       help='每个 episode 放置的 workflow 数量')
    parser.add_argument('--num_episodes', type=int, default=400,
                       help='Episode 数量（重复次数）')
    parser.add_argument('--test_mode', action='store_true',
                       help='启用测试模式，打印放置前标签和放置动作')
    parser.add_argument('--test_episode_idx', type=int, default=0,
                       help='测试模式下要打印的 episode 索引（默认 0）')
    parser.add_argument('--bias', type=float, default=0.3,
                       help='bias 参数（约束节点 bias_cpu = bias * max_cpu）')
    parser.add_argument('--noderank_iterations', type=int, default=3,
                       help='NodeRank 迭代次数（默认 2）')
    
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
        test_episode_idx=args.test_episode_idx,
        bias=args.bias,
        noderank_iterations=args.noderank_iterations
    )


if __name__ == '__main__':
    main()
