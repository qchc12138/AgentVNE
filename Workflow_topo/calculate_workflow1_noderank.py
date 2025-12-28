import json
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# 添加父目录到路径，以便导入 noderank_calculator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noderank_calculator import VNNodeRankCalculator, calculate_vn_noderank

PJ_u = 0.80
PF_u = 0.20

def load_topology(file_path: str) -> Tuple[List[Dict], List[Dict], bool]:
    """
    加载拓扑结构数据
    
    Args:
        file_path: 拓扑文件路径
    
    Returns:
        nodes: 节点列表
        links: 链路列表
        directed: 是否为有向图
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['nodes'], data['links'], data.get('directed', False)


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
    H(u) = CPU(u) * comm_bandwidth
    
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
        comm_bw = node['comm_bandwidth']
        
        # 计算所有出链的带宽总和
        total_outgoing_bw = sum(link['bandwidth'] for link in outgoing_links[i])
        
        # H(u) = CPU(u) * comm_bandwidth
        H[i] = cpu * comm_bw
        
        print(f"节点 {i}:")
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
    pF_uv = H(v) / Σ H(w)，其中 w ∈ nbr1(u)
    
    Args:
        H: 每个节点的资源评估值
        adjacency: 邻接信息
    
    Returns:
        pF: 前向概率矩阵 (n × n)
    """
    n = len(H)
    pF = np.zeros((n, n))
    
    for u in range(n):
        # 邻居节点集合
        neighbors = adjacency[u]
        
        if len(neighbors) == 0:
            # 如果没有邻居，跳过
            continue
        
        # 计算邻居节点的资源评估总和
        neighbor_H_sum = sum(H[w] for w in neighbors)
        
        # 计算到每个邻居的前向概率
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
                       tolerance: float = 1e-6) -> Tuple[np.ndarray, int]:
    """
    步骤4：NodeRank 的计算（只迭代一次）
    NR^(1)(v) = pJ_u × NR^(0)(v) + pF_u × Σ pF_uv × NR^(0)(u)
    
    Args:
        NR_0: 初始 NodeRank 值
        pJ: 跳转概率矩阵（未使用）
        pF: 前向概率矩阵
        pJ_u: 跳转偏置因子（默认 0.80）
        pF_u: 前向偏置因子（默认 0.20）
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
    print()
    
    NR = NR_0.copy()
    
    # 只迭代一次
    # NR^(1)(v) = pJ_u × NR^(0)(v) + pF_u × Σ pF_uv × NR^(0)(u)
    NR_new = pJ_u * NR + pF_u * (pF @ NR)
    
    # 归一化（确保总和为1）
    NR_new = NR_new / np.sum(NR_new)
    
    print("迭代 1: 完成")
    for i in range(len(NR_new)):
        print(f"  节点 {i}: NR^(1)({i}) = {NR_new[i]:.8f}")
    
    return NR_new, 1


def calculate_noderank(topology_file: str, 
                       pJ_u: float = PJ_u, 
                       pF_u: float = PF_u,
                       max_iterations: int = 100,
                       tolerance: float = 1e-6) -> Dict:
    """
    完整的 NodeRank 计算流程
    
    Args:
        topology_file: 拓扑文件路径
        pJ_u: 跳转偏置因子
        pF_u: 前向偏置因子
        max_iterations: 最大迭代次数
        tolerance: 收敛容忍度
    
    Returns:
        包含计算结果的字典
    """
    print("开始计算 NodeRank...")
    print()
    
    # 加载拓扑
    nodes, links, directed = load_topology(topology_file)
    n = len(nodes)
    
    print(f"拓扑信息:")
    print(f"  节点数: {n}")
    print(f"  链路数: {len(links)}")
    print(f"  图类型: {'有向图' if directed else '无向图'}")
    print()
    
    # 构建邻接信息
    adj_info = build_adjacency_info(nodes, links, directed=directed)
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
    NR_final, iterations = iterative_noderank(NR_0, pJ, pF, pJ_u, pF_u, max_iterations, tolerance)
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("最终 NodeRank 结果")
    print("=" * 60)
    
    # 按 NodeRank 值排序
    node_ranks = [(i, NR_final[i]) for i in range(n)]
    node_ranks.sort(key=lambda x: x[1], reverse=True)
    
    print("\n按 NodeRank 值排序（从高到低）:")
    for rank, (node_id, nr_value) in enumerate(node_ranks, 1):
        print(f"排名 {rank}: 节点 {node_id} - NodeRank = {nr_value:.8f}")
    
    return {
        'noderank': NR_final.tolist(),
        'initial_noderank': NR_0.tolist(),
        'resource_evaluation': H.tolist(),
        'iterations': iterations,
        'nodes': nodes,
        'adjacency': adjacency,
        'directed': directed
    }


def save_results(results: Dict, output_file: str):
    """
    保存计算结果到JSON文件
    
    Args:
        results: 计算结果字典
        output_file: 输出文件路径
    """
    # 准备要保存的数据
    output_data = {
        'noderank': results['noderank'],
        'initial_noderank': results['initial_noderank'],
        'resource_evaluation': results['resource_evaluation'],
        'iterations': results['iterations'],
        'directed': results['directed']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    # 计算 workflow1_topo 的 NodeRank
    topology_file = '/home/yc2/mrt/a/workflow_topo/workflow1_topo.json'
    output_file = '/home/yc2/mrt/a/workflow_topo/workflow1_noderank.json'
    
    # 方式1：使用类（推荐，可在其他文件中调用）
    calculator = VNNodeRankCalculator(
        pJ_u=PJ_u,
        pF_u=PF_u,
        verbose=True
    )
    results = calculator.calculate(topology_file)
    
    # 方式2：使用便捷函数
    # results = calculate_vn_noderank(
    #     topology_file=topology_file,
    #     pJ_u=PJ_u,
    #     pF_u=PF_u,
    #     verbose=True
    # )
    
    # 方式3：使用原有函数（保持向后兼容）
    # results = calculate_noderank(
    #     topology_file=topology_file,
    #     pJ_u=PJ_u,
    #     pF_u=PF_u,
    #     max_iterations=100,
    #     tolerance=1e-6
    # )
    
    # 保存结果
    save_results(results, output_file)
    
    print("\n" + "=" * 60)
    print("计算完成！")
    print("=" * 60)

