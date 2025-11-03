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
from typing import List, Dict, Tuple
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


def _nodes_to_features(nodes: List[Dict]) -> torch.Tensor:
    """将节点属性映射为 6 维特征 [type, memory, cpu, disk, comm_bw, num_nodes]。"""
    num_nodes = len(nodes)
    feats: List[List[float]] = []
    for n in nodes:
        node_type = n.get('node_type', 'unknown')
        type_val = 1.0 if isinstance(node_type, str) and node_type.lower() == 'edge' else 0.0
        memory = float(n.get('memory', 0.0))
        cpu = float(n.get('cpu', 0.0))
        disk = float(n.get('disk', 0.0))
        comm_bw = float(n.get('comm_bandwidth', n.get('bandwidth', 0.0)))
        feats.append([type_val, memory, cpu, disk, comm_bw, float(num_nodes)])
    return torch.tensor(feats, dtype=torch.float)


def _topology_to_pyg_data(topo: Dict) -> Data:
    """将拓扑 dict 转为 torch_geometric.data.Data。
    需要字段: topo['nodes'] (带资源/需求), topo['links'] (source/target)。
    使用 topo['directed'] 判断是否有向。
    """
    nodes = topo['nodes']
    links = topo.get('links', [])
    directed = bool(topo.get('directed', False))
    x = _nodes_to_features(nodes)
    edge_index = _build_edge_index(links, directed=directed)
    return Data(x=x, edge_index=edge_index)


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


def _greedy_place_workflow(
    workflow_topo: Dict,
    sn_nodes: List[Dict],
    sn_noderank: np.ndarray,
    workflow_noderank: List[float]
) -> None:
    """将一个 workflow 的所有节点贪心放置到底层网络，并原地更新 sn_nodes 的 cpu/memory/disk。
    放置顺序：按 workflow_noderank 从高到低选 workflow 节点；为其在 SN 中按 sn_noderank 从高到低选能满足资源的节点。
    仅检查并扣减 cpu、memory、disk。
    """
    # 排序索引
    wf_nodes = workflow_topo['nodes']
    N1 = len(wf_nodes)
    wf_order = list(sorted(range(N1), key=lambda i: workflow_noderank[i], reverse=True))
    sn_order = list(sorted(range(len(sn_nodes)), key=lambda j: sn_noderank[j], reverse=True))

    for i in wf_order:
        demand_cpu = float(wf_nodes[i].get('cpu', 0.0))
        demand_mem = float(wf_nodes[i].get('memory', 0.0))
        demand_disk = float(wf_nodes[i].get('disk', 0.0))

        placed = False
        for j in sn_order:
            cap_cpu = float(sn_nodes[j].get('cpu', 0.0))
            cap_mem = float(sn_nodes[j].get('memory', 0.0))
            cap_disk = float(sn_nodes[j].get('disk', 0.0))
            if cap_cpu >= demand_cpu and cap_mem >= demand_mem and cap_disk >= demand_disk:
                # 扣减资源
                sn_nodes[j]['cpu'] = cap_cpu - demand_cpu
                sn_nodes[j]['memory'] = cap_mem - demand_mem
                sn_nodes[j]['disk'] = cap_disk - demand_disk
                placed = True
                break
        # 若无法放置，按指示仅跳过（不放置该节点）
        if not placed:
            continue


def generate_pretrain_dataset(
    sn_topo_path: str,
    workflow_topo_path: str,
    workflow_noderank_path: str,
    output_path: str,
    test_output_path: str = None,
    workflows_per_episode: int = 10,
    num_episodes: int = 50
) -> None:
    """生成预训练数据集并保存到文件。
    
    Args:
        sn_topo_path: 底层网络拓扑文件路径
        workflow_topo_path: Workflow 拓扑文件路径
        workflow_noderank_path: Workflow NodeRank 文件路径
        output_path: 输出数据集文件路径（.pt 格式）
        workflows_per_episode: 每个 episode 放置的 workflow 数量
        num_episodes: episode 数量（重复次数）
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
    
    print(f"  底层网络节点数: {len(base_sn_topo['nodes'])}")
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
        test_workflow_graph = _topology_to_pyg_data(base_workflow_topo)
        test_substrate_graph = _topology_to_pyg_data(base_sn_topo)
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
                'sn_topo_path': sn_topo_path,
                'workflow_topo_path': workflow_topo_path,
                'workflow_noderank_path': workflow_noderank_path
            }
        }, test_output_path)
        print(f"测试样本已保存到: {test_output_path}")

    # 生成训练样本
    samples: List[Dict] = []
    total_samples = num_episodes * workflows_per_episode
    
    print(f"\n开始生成 {total_samples} 个样本...")
    with tqdm(total=total_samples, desc="生成样本") as pbar:
        for episode_idx in range(num_episodes):
            # 重置 SN 到初始状态（深拷贝）
            sn_topo = json.loads(json.dumps(base_sn_topo))
            
            for wf_idx in range(workflows_per_episode):
                # 计算当前 SN 的 noderank
                sn_noderank = _compute_sn_noderank(
                    sn_topo['nodes'], 
                    sn_topo.get('links', []), 
                    directed=bool(sn_topo.get('directed', False))
                )
                
                # x: 当前的 workflow 与 SN 图（放置前状态）
                workflow_graph = _topology_to_pyg_data(base_workflow_topo)
                substrate_graph = _topology_to_pyg_data(sn_topo)
                
                # y: 将 SN 的 noderank 重复 N1 行
                N1 = workflow_graph.x.size(0)
                N2 = substrate_graph.x.size(0)
                y = torch.tensor(
                    np.tile(sn_noderank.reshape(1, N2), (N1, 1)), 
                    dtype=torch.float
                )
                
                # 保存样本
                samples.append({
                    'workflow_graph': workflow_graph,
                    'substrate_graph': substrate_graph,
                    'label': y
                })
                
                pbar.update(1)
                
                # 放置该 workflow 并更新 SN 资源
                _greedy_place_workflow(
                    base_workflow_topo, 
                    sn_topo['nodes'], 
                    sn_noderank, 
                    workflow_noderank
                )
    
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
                       default='/home/zrz/SimuVNE/topo/SN_topology.json',
                       help='底层网络拓扑文件路径')
    parser.add_argument('--workflow_topo', type=str,
                       default='/home/zrz/SimuVNE/workflow_topo/workflow1_topo.json',
                       help='Workflow 拓扑文件路径')
    parser.add_argument('--workflow_noderank', type=str,
                       default='/home/zrz/SimuVNE/workflow_topo/workflow1_noderank.json',
                       help='Workflow NodeRank 文件路径')
    parser.add_argument('--output', type=str,
                       default='/home/zrz/SimuVNE/pretrain_data/pretrain_dataset.pt',
                       help='输出数据集文件路径')
    parser.add_argument('--test_output', type=str,
                       default='/home/zrz/SimuVNE/pretrain_data/test_sample.pt',
                       help='测试样本输出文件路径（单条）')
    parser.add_argument('--workflows_per_episode', type=int, default=10,
                       help='每个 episode 放置的 workflow 数量')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Episode 数量（重复次数）')
    
    args = parser.parse_args()
    
    generate_pretrain_dataset(
        sn_topo_path=args.sn_topo,
        workflow_topo_path=args.workflow_topo,
        workflow_noderank_path=args.workflow_noderank,
        output_path=args.output,
        test_output_path=args.test_output,
        workflows_per_episode=args.workflows_per_episode,
        num_episodes=args.num_episodes
    )


if __name__ == '__main__':
    main()
