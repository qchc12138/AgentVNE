import json
import numpy as np
from typing import Dict, List, Tuple

PJ_u = 0.80
PF_u = 0.20

def load_topology(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    加载拓扑结构数据
    
    Args:
        file_path: 拓扑文件路径
    
    Returns:
        nodes: 节点列表（已过滤掉位置特征 x, y, pos）
        links: 链路列表
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 过滤掉节点的位置特征 (x, y, pos)，只保留必要的资源特征
    filtered_nodes = []
    for node in data['nodes']:
        filtered_node = {
            'cpu': node['cpu'],
            'memory': node.get('memory', 0),
            'disk': node.get('disk', 0),
            'bandwidth': node.get('bandwidth', 0),
            'comm_bandwidth': node.get('comm_bandwidth', 0),
            'node_type': node.get('node_type', 'unknown'),
        }
        # 保留其他可能需要的属性
        for key in ['id']:
            if key in node:
                filtered_node[key] = node[key]
        filtered_nodes.append(filtered_node)
    
    return filtered_nodes, data['links']


def build_adjacency_info(nodes: List[Dict], links: List[Dict], directed: bool = False) -> Dict:
    """
    构建邻接信息
    
    Args:
        nodes: 节点列表
        links: 链路列表
        directed: 是否为有向图
    
    Returns:
        包含邻接信息的字典
    """
    n = len(nodes)
    adjacency = {i: [] for i in range(n)}
    outgoing_links = {i: [] for i in range(n)}
    
    for link in links:
        source = link['source']
        target = link['target']
        bw = link['bandwidth']
        
        # 添加出链信息
        outgoing_links[source].append({'target': target, 'bandwidth': bw})
        
        if not directed:
            # 无向图，反向也是出链
            outgoing_links[target].append({'target': source, 'bandwidth': bw})
        
        # 邻接关系
        if target not in adjacency[source]:
            adjacency[source].append(target)
        if not directed and source not in adjacency[target]:
            adjacency[target].append(source)
    
    return {'adjacency': adjacency, 'outgoing_links': outgoing_links}


def calculate_initial_resource_evaluation(nodes: List[Dict], 
                                          outgoing_links: Dict[int, List[Dict]]) -> np.ndarray:
    """
    步骤1：计算每个节点的初始资源评估
    H(u) = CPU(u) * Σ BW(l)，其中 l ∈ L(u)
    
    Args:
        nodes: 节点列表
        outgoing_links: 每个节点的出链信息
    
    Returns:
        H: 每个节点的资源评估值数组
    """
    n = len(nodes)
    H = np.zeros(n)
    
    print("=" * 60)
    print("步骤1：计算每个节点的初始资源评估")
    print("=" * 60)
    
    for i, node in enumerate(nodes):
        cpu = node['cpu']
        # comm_bandwidth 是节点的通信带宽属性
        comm_bw = node['comm_bandwidth']
        
        # 计算所有出链的带宽总和
        total_outgoing_bw = sum(link['bandwidth'] for link in outgoing_links[i])
        
        # H(u) = CPU(u) * comm_bandwidth
        # 根据用户要求："为cpu核心数*通信带宽"
        H[i] = cpu * comm_bw
        
        print(f"节点 {i} ({node.get('node_type', 'unknown')}):")
        print(f"  CPU: {cpu}")
        print(f"  通信带宽 (comm_bandwidth): {comm_bw}")
        print(f"  出链总带宽: {total_outgoing_bw}")
        print(f"  H({i}) = {cpu} × {comm_bw} = {H[i]}")
        print()
    
    return H


def calculate_initial_noderank(H: np.ndarray) -> np.ndarray:
    """
    步骤2：计算初始 NodeRank 值
    NR^(0)(u) = H(u) / Σ H(v)，其中 v ∈ V
    
    Args:
        H: 每个节点的资源评估值
    
    Returns:
        NR_0: 初始 NodeRank 值
    """
    print("=" * 60)
    print("步骤2：计算初始 NodeRank 值")
    print("=" * 60)
    
    total_H = np.sum(H)
    NR_0 = H / total_H
    
    print(f"所有节点的资源评估总和: Σ H(v) = {total_H}")
    print()
    for i in range(len(NR_0)):
        print(f"节点 {i}: NR^(0)({i}) = {H[i]}/{total_H} = {NR_0[i]:.6f}")
    print()
    
    return NR_0


def calculate_jumping_probability(H: np.ndarray) -> np.ndarray:
    """
    步骤3a：计算跳转概率矩阵
    pJ_uv = H(v) / Σ H(w)，其中 w ∈ V
    
    Args:
        H: 每个节点的资源评估值
    
    Returns:
        pJ: 跳转概率矩阵 (n × n)
    """
    total_H = np.sum(H)
    n = len(H)
    pJ = np.zeros((n, n))
    
    for u in range(n):
        for v in range(n):
            pJ[u][v] = H[v] / total_H
    
    return pJ


def calculate_forward_probability(H: np.ndarray, adjacency: Dict[int, List[int]]) -> np.ndarray:
    """
    步骤3b：计算前向概率矩阵
    pF_uv = H(v) / Σ H(w)，其中 w ∈ nbr1(u) ∪ {u}（包括节点自身）
    
    Args:
        H: 每个节点的资源评估值
        adjacency: 邻接信息
    
    Returns:
        pF: 前向概率矩阵 (n × n)
    """
    n = len(H)
    pF = np.zeros((n, n))
    
    for u in range(n):
        # 邻居节点集合，包括节点自己
        # neighbors = adjacency[u] + [u]  # 添加节点自己
        neighbors = adjacency[u] # 不添加节点自己
        
        # 计算邻居节点（包括自己）的资源评估总和
        neighbor_H_sum = sum(H[w] for w in neighbors)
        
        # 计算到每个邻居（包括自己）的前向概率
        for v in neighbors:
            pF[u][v] = H[v] / neighbor_H_sum
    
    return pF


def print_probability_matrices(pJ: np.ndarray, pF: np.ndarray):
    """打印概率矩阵"""
    print("=" * 60)
    print("步骤3：计算两种概率")
    print("=" * 60)
    
    print("\n跳转概率矩阵 pJ (每行代表从节点u到所有节点v的跳转概率):")
    print("     ", end="")
    for i in range(len(pJ)):
        print(f"  节点{i}  ", end="")
    print()
    for i in range(len(pJ)):
        print(f"节点{i} ", end="")
        for j in range(len(pJ)):
            print(f" {pJ[i][j]:.4f} ", end="")
        print()
    
    print("\n前向概率矩阵 pF (每行代表从节点u到其邻居v的前向概率):")
    print("     ", end="")
    for i in range(len(pF)):
        print(f"  节点{i}  ", end="")
    print()
    for i in range(len(pF)):
        print(f"节点{i} ", end="")
        for j in range(len(pF)):
            if pF[i][j] > 0:
                print(f" {pF[i][j]:.4f} ", end="")
            else:
                print(f"  0.0000 ", end="")
        print()
    print()


def iterative_noderank(NR_0: np.ndarray, 
                       pJ: np.ndarray, 
                       pF: np.ndarray,
                       pJ_u: float = PJ_u,
                       pF_u: float = PF_u,
                       max_iterations: int = 100,
                       tolerance: float = 1e-6,
                       num_first_transforms: int = 2) -> Tuple[np.ndarray, int]:
    """
    步骤4：NodeRank 的计算（只迭代一次）
    NR^(1)(v) = pJ_u × NR^(0)(v) + pF_u × Σ pF_uv × NR^(0)(u)
    
    Args:
        NR_0: 初始 NodeRank 值
        pJ: 跳转概率矩阵（未使用）
        pF: 前向概率矩阵
        pJ_u: 跳转偏置因子（默认 0.85）
        pF_u: 前向偏置因子（默认 0.15）
        max_iterations: 最大迭代次数（未使用，保留用于接口兼容）
        tolerance: 收敛容忍度（未使用，保留用于接口兼容）
    
    Returns:
        NR_final: 最终的 NodeRank 值
        iterations: 实际迭代次数（总是返回1）
    """
    print("=" * 60)
    print("步骤4：NodeRank 迭代计算")
    print("=" * 60)
    print(f"偏置因子: pJ_u = {pJ_u}, pF_u = {pF_u}")
    print("只进行一次迭代计算")
    print(f"第一次变换重复次数: {num_first_transforms}")
    print()
    
    NR = NR_0.copy()
    
    # 只迭代一次
    # 原逻辑（已注释）：
    # NR^(1)(v) = pJ_u × NR^(0)(v) + pF_u × Σ pF_uv × NR^(0)(u)
    # NR_new = pJ_u * NR + pF_u * (pF @ NR)
    # NR_new = NR_new / np.sum(NR_new)

    # 新逻辑：两次变换
    # 1) 第一次变换（可重复）：节点的 noderank = 自身的 noderank + 邻居节点的 noderank，然后全体归一化
    # 利用 pF 的非零元素表示邻居关系（不包含自身）
    neighbor_mask = (pF > 0).astype(np.float64)
    # 旧实现（固定两次，已注释）
    # NR_step1 = NR + pF_u * (pF @ NR)
    # NR_step1 = NR_step1 / np.sum(NR_step1)
    # NR_step2 = NR_step1 + pF_u * (pF @ NR_step1)
    # NR_step2 = NR_step2 / np.sum(NR_step2)
    # NR_new = NR_step2

    NR_current = NR
    num_first_transforms = 2
    for _ in range(num_first_transforms):
        # 自身 + 邻居求和（通过乘以 pF_u 控制邻居贡献强度），随后归一化
        NR_next = NR_current + pF_u * (pF @ NR_current)
        NR_current = NR_next / np.sum(NR_next)

    # 2) 第二次变换：对所有节点的 noderank 做 softmax（如需启用，取消注释）
    # exp_scores = np.exp(NR_current - np.max(NR_current))
    # NR_new = exp_scores / np.sum(exp_scores)
    
    # 接下来对NR_current平方再归一化得到NR_new
    NR_new = NR_current ** 3
    NR_new = NR_new / np.sum(NR_new)
    
    print("迭代 1: 完成")
    for i in range(len(NR_new)):
        print(f"  节点 {i}: NR^(1)({i}) = {NR_new[i]:.8f}")
    
    return NR_new, 1


def calculate_noderank(topology_file: str, 
                       pJ_u: float = PJ_u, 
                       pF_u: float = PF_u,
                       max_iterations: int = 100,
                       tolerance: float = 1e-6,
                       num_first_transforms: int = 2) -> Dict:
    """
    完整的 NodeRank 计算流程
    
    Args:
        topology_file: 拓扑文件路径
        pJ_u: 跳转偏置因子
        pF_u: 前向偏置因子
        max_iterations: 最大迭代次数
        tolerance: 收敛容忍度
        num_first_transforms: 第一次变换的重复次数
    
    Returns:
        包含计算结果的字典
    """
    print("开始计算 NodeRank...")
    print()
    
    # 加载拓扑
    nodes, links = load_topology(topology_file)
    n = len(nodes)
    
    print(f"拓扑信息:")
    print(f"  节点数: {n}")
    print(f"  链路数: {len(links)}")
    print()
    
    # 构建邻接信息
    adj_info = build_adjacency_info(nodes, links, directed=False)
    adjacency = adj_info['adjacency']
    outgoing_links = adj_info['outgoing_links']
    
    # 步骤1：计算初始资源评估
    H = calculate_initial_resource_evaluation(nodes, outgoing_links)
    
    # 步骤2：计算初始 NodeRank
    NR_0 = calculate_initial_noderank(H)
    
    # 步骤3：计算概率矩阵
    pJ = calculate_jumping_probability(H)
    pF = calculate_forward_probability(H, adjacency)
    print_probability_matrices(pJ, pF)
    
    # 步骤4：迭代计算 NodeRank
    NR_final, iterations = iterative_noderank(NR_0, pJ, pF, pJ_u, pF_u, max_iterations, tolerance, num_first_transforms)
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("最终 NodeRank 结果")
    print("=" * 60)
    
    # 按 NodeRank 值排序
    node_ranks = [(i, NR_final[i], nodes[i].get('node_type', 'unknown')) 
                  for i in range(n)]
    node_ranks.sort(key=lambda x: x[1], reverse=True)
    
    print("\n按 NodeRank 值排序（从高到低）:")
    for rank, (node_id, nr_value, node_type) in enumerate(node_ranks, 1):
        print(f"排名 {rank}: 节点 {node_id} ({node_type}) - NodeRank = {nr_value:.8f}")
    
    return {
        'noderank': NR_final,
        'initial_noderank': NR_0,
        'resource_evaluation': H,
        'iterations': iterations,
        'nodes': nodes,
        'adjacency': adjacency
    }


if __name__ == '__main__':
    # 计算 star_topology 的 NodeRank
    topology_file = '/home/zrz/SimuVNE/topo/SN_topology.json'
    
    results = calculate_noderank(
        topology_file=topology_file,
        pJ_u=PJ_u,      # 跳转偏置因子
        pF_u=PF_u,      # 前向偏置因子
        max_iterations=100,
        tolerance=1e-6
    )
    
    # 保存 NodeRank 结果为 JSON 文件
    output_file = topology_file.replace('.json', '_noderank.json')
    
    # 准备要保存的数据
    output_data = {
        'topology_file': topology_file,
        'parameters': {
            'pJ_u': PJ_u,
            'pF_u': PF_u
        },
        'noderank': results['noderank'].tolist(),
        'initial_noderank': results['initial_noderank'].tolist(),
        'resource_evaluation': results['resource_evaluation'].tolist(),
        'iterations': results['iterations'],
        'node_details': []
    }
    
    # 添加每个节点的详细信息
    nodes = results['nodes']
    for i, node in enumerate(nodes):
        node_info = {
            'node_id': i,
            'node_type': node.get('node_type', 'unknown'),
            'cpu': node.get('cpu', 0),
            'comm_bandwidth': node.get('comm_bandwidth', 0),
            'noderank': float(results['noderank'][i]),
            'initial_noderank': float(results['initial_noderank'][i]),
            'resource_evaluation': float(results['resource_evaluation'][i])
        }
        output_data['node_details'].append(node_info)
    
    # 按 NodeRank 值排序
    output_data['node_details'].sort(key=lambda x: x['noderank'], reverse=True)
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("计算完成！")
    print(f"NodeRank 结果已保存到: {output_file}")
    print("=" * 60)

