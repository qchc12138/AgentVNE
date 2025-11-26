#!/usr/bin/env python3
"""
拓扑生成模块
实现各种网络拓扑生成算法，包括 Waxman 模型
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional


def generate_waxman_topology(
    num_nodes: int,
    alpha: float = 0.5,
    beta: float = 0.2,
    domain_size: Tuple[float, float] = (100, 100),
    seed: Optional[int] = None
) -> nx.Graph:
    """
    使用 Waxman 模型生成网络拓扑
    
    Waxman 模型通过概率方式生成边：P(u,v) = β * exp(-d/(α*L))
    - d 是节点 u 和 v 之间的欧几里得距离
    - L 是所有节点对之间的最大距离
    - α 控制距离对连接概率的影响程度（越大，远距离连接越容易）
    - β 控制整体的边密度（越大，图越密集）
    
    这个模型能够模拟真实网络的特性：
    - 局部连接密集（近距离节点更容易连接）
    - 远距离连接稀疏（远距离节点连接概率低）
    
    参数:
        num_nodes: 节点数量
        alpha: α 参数，控制距离的影响 (0 < α <= 1)
        beta: β 参数，控制边的密度 (0 < β <= 1)
        domain_size: 节点分布的二维空间大小 (width, height)
        seed: 随机种子，用于结果复现
    
    返回:
        NetworkX Graph 对象，包含节点位置信息
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 创建空图
    G = nx.Graph()
    
    # 在二维平面上随机放置节点
    positions = {}
    for i in range(num_nodes):
        x = np.random.uniform(0, domain_size[0])
        y = np.random.uniform(0, domain_size[1])
        positions[i] = (x, y)
        G.add_node(i, pos=(x, y), x=x, y=y)
    
    # 计算所有节点对之间的距离
    distances = {}
    max_distance = 0.0
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pos_i = positions[i]
            pos_j = positions[j]
            # 计算欧几里得距离
            distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
            distances[(i, j)] = distance
            max_distance = max(max_distance, distance)
    
    # 防止除零错误
    if max_distance == 0:
        max_distance = 1.0
    
    # 根据 Waxman 概率模型添加边
    for (i, j), distance in distances.items():
        # 计算连接概率: P(u,v) = β * exp(-d/(α*L))
        probability = beta * np.exp(-distance / (alpha * max_distance))
        
        # 根据概率决定是否添加边
        if np.random.random() < probability:
            G.add_edge(i, j, weight=distance)
    
    # 确保图是连通的（如果不连通，添加最小生成树的边）
    if not nx.is_connected(G):
        G = _ensure_connected(G, positions)
    
    return G


def _ensure_connected(G: nx.Graph, positions: dict) -> nx.Graph:
    """
    确保图是连通的
    如果图不连通，通过添加最短边连接各个连通分量
    
    参数:
        G: 原始图
        positions: 节点位置字典
    
    返回:
        连通的图
    """
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    
    if len(components) <= 1:
        return G
    
    # 将连通分量转换为列表，以便索引
    components = [list(comp) for comp in components]
    
    # 连接相邻的连通分量
    for i in range(len(components) - 1):
        # 找到两个连通分量之间距离最短的边
        min_distance = float('inf')
        min_edge = None
        
        for node_i in components[i]:
            for node_j in components[i + 1]:
                pos_i = positions[node_i]
                pos_j = positions[node_j]
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    min_edge = (node_i, node_j)
        
        # 添加最短边
        if min_edge:
            G.add_edge(min_edge[0], min_edge[1], weight=min_distance)
    
    return G


def generate_topology(
    num_nodes: int,
    topology_type: str = 'waxman',
    **kwargs
) -> nx.Graph:
    """
    通用拓扑生成接口
    
    参数:
        num_nodes: 节点数量
        topology_type: 拓扑类型，支持 'waxman', 'erdos_renyi', 'barabasi_albert' 等
        **kwargs: 特定拓扑模型的参数
    
    返回:
        NetworkX Graph 对象
    """
    if topology_type.lower() == 'waxman':
        alpha = kwargs.get('alpha', 0.5)
        beta = kwargs.get('beta', 0.2)
        domain_size = kwargs.get('domain_size', (100, 100))
        seed = kwargs.get('seed', None)
        return generate_waxman_topology(num_nodes, alpha, beta, domain_size, seed)
    
    elif topology_type.lower() == 'erdos_renyi' or topology_type.lower() == 'er':
        # Erdős-Rényi 随机图
        p = kwargs.get('p', 0.3)
        seed = kwargs.get('seed', None)
        return nx.erdos_renyi_graph(num_nodes, p, seed=seed)
    
    elif topology_type.lower() == 'barabasi_albert' or topology_type.lower() == 'ba':
        # Barabási-Albert 无标度网络
        m = kwargs.get('m', 2)
        seed = kwargs.get('seed', None)
        return nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    
    elif topology_type.lower() == 'watts_strogatz' or topology_type.lower() == 'ws':
        # Watts-Strogatz 小世界网络
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.3)
        seed = kwargs.get('seed', None)
        return nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
    
    else:
        raise ValueError(f"不支持的拓扑类型: {topology_type}")


def print_topology_info(G: nx.Graph):
    """
    打印拓扑信息
    
    参数:
        G: NetworkX Graph 对象
    """
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"是否连通: {nx.is_connected(G)}")
    
    if nx.is_connected(G):
        print(f"平均最短路径长度: {nx.average_shortest_path_length(G):.2f}")
        print(f"直径: {nx.diameter(G)}")
    
    print(f"聚类系数: {nx.average_clustering(G):.4f}")


# 示例使用
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Waxman 拓扑生成示例")
    print("=" * 60)
    
    # 生成 Waxman 拓扑
    num_nodes = 20
    alpha = 0.5
    beta = 0.2
    
    print(f"\n生成参数:")
    print(f"  节点数: {num_nodes}")
    print(f"  α (alpha): {alpha}")
    print(f"  β (beta): {beta}")
    print()
    
    G = generate_waxman_topology(
        num_nodes=num_nodes,
        alpha=alpha,
        beta=beta,
        seed=42
    )
    
    print("拓扑信息:")
    print_topology_info(G)
    
    # 可视化
    print("\n绘制拓扑图...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制网络
        nx.draw(
            G, pos,
            ax=ax,
            with_labels=True,
            node_color='lightblue',
            node_size=500,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=1.5,
            alpha=0.7
        )
        
        ax.set_title(f'Waxman 拓扑 (α={alpha}, β={beta}, n={num_nodes})')
        ax.axis('off')
        plt.tight_layout()
        
        # 保存图像
        output_path = '/home/yc2/mrt/a/topo/waxman_topology.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"拓扑图已保存到: {output_path}")
    except Exception as e:
        print(f"绘制拓扑图时出现错误: {e}")
        print("拓扑生成成功，但跳过可视化步骤")
    
    print("\n完成！")

