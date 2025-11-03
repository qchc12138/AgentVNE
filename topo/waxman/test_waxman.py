#!/usr/bin/env python3
"""
Waxman 拓扑生成测试脚本
演示如何使用拓扑生成模块
"""

from topo_create import generate_waxman_topology, print_topology_info
import networkx as nx


def test_basic_generation():
    """测试基本拓扑生成"""
    print("=" * 60)
    print("测试 1: 基本 Waxman 拓扑生成")
    print("=" * 60)
    
    G = generate_waxman_topology(
        num_nodes=20,
        alpha=0.5,
        beta=0.2,
        seed=42
    )
    
    print_topology_info(G)
    print()


def test_parameter_variations():
    """测试不同参数配置"""
    print("=" * 60)
    print("测试 2: 不同参数配置的影响")
    print("=" * 60)
    
    configs = [
        (0.5, 0.2, "标准配置（稀疏）"),
        (0.5, 0.4, "增加边密度"),
        (0.8, 0.2, "增加远距离连接"),
        (0.3, 0.5, "高密度但偏向近距离"),
    ]
    
    num_nodes = 30
    
    for alpha, beta, description in configs:
        print(f"\n{description} (α={alpha}, β={beta}):")
        G = generate_waxman_topology(
            num_nodes=num_nodes,
            alpha=alpha,
            beta=beta,
            seed=42
        )
        
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees)
        
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {G.number_of_edges()}")
        print(f"  平均度: {avg_degree:.2f}")
        print(f"  边密度: {nx.density(G):.4f}")
    
    print()


def test_network_sizes():
    """测试不同规模的网络"""
    print("=" * 60)
    print("测试 3: 不同规模的网络生成")
    print("=" * 60)
    
    sizes = [10, 20, 50, 100]
    alpha, beta = 0.5, 0.2
    
    for size in sizes:
        print(f"\n生成 {size} 节点的网络...")
        G = generate_waxman_topology(
            num_nodes=size,
            alpha=alpha,
            beta=beta,
            seed=42
        )
        
        print(f"  边数: {G.number_of_edges()}")
        print(f"  平均度: {sum(dict(G.degree()).values()) / size:.2f}")
        
        if nx.is_connected(G):
            print(f"  平均路径长度: {nx.average_shortest_path_length(G):.2f}")
    
    print()


def test_node_attributes():
    """测试节点属性"""
    print("=" * 60)
    print("测试 4: 节点位置属性")
    print("=" * 60)
    
    G = generate_waxman_topology(
        num_nodes=10,
        alpha=0.5,
        beta=0.2,
        domain_size=(100, 100),
        seed=42
    )
    
    print("\n前5个节点的位置信息:")
    for node in range(min(5, G.number_of_nodes())):
        pos = G.nodes[node]['pos']
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        print(f"  节点 {node}: pos={pos}, x={x:.2f}, y={y:.2f}")
    
    # 计算节点间距离
    if G.number_of_nodes() >= 2:
        pos_0 = G.nodes[0]['pos']
        pos_1 = G.nodes[1]['pos']
        import math
        distance = math.sqrt((pos_0[0] - pos_1[0])**2 + (pos_0[1] - pos_1[1])**2)
        print(f"\n节点0和节点1之间的距离: {distance:.2f}")
    
    print()


def test_reproducibility():
    """测试随机种子的可复现性"""
    print("=" * 60)
    print("测试 5: 随机种子可复现性")
    print("=" * 60)
    
    # 使用相同的种子生成两次
    G1 = generate_waxman_topology(num_nodes=20, alpha=0.5, beta=0.2, seed=42)
    G2 = generate_waxman_topology(num_nodes=20, alpha=0.5, beta=0.2, seed=42)
    
    # 比较
    same_nodes = G1.number_of_nodes() == G2.number_of_nodes()
    same_edges = G1.number_of_edges() == G2.number_of_edges()
    same_structure = nx.is_isomorphic(G1, G2)
    
    print(f"\n相同种子 (seed=42) 生成两次:")
    print(f"  节点数相同: {same_nodes}")
    print(f"  边数相同: {same_edges}")
    print(f"  结构同构: {same_structure}")
    
    # 使用不同种子
    G3 = generate_waxman_topology(num_nodes=20, alpha=0.5, beta=0.2, seed=123)
    different_structure = not nx.is_isomorphic(G1, G3)
    
    print(f"\n不同种子 (seed=42 vs seed=123):")
    print(f"  结构不同: {different_structure}")
    print(f"  G1 边数: {G1.number_of_edges()}, G3 边数: {G3.number_of_edges()}")
    
    print()


def test_connectivity():
    """测试连通性保证"""
    print("=" * 60)
    print("测试 6: 连通性保证")
    print("=" * 60)
    
    # 使用较小的 beta 值，容易产生不连通的图
    num_tests = 10
    all_connected = True
    
    print(f"\n生成 {num_tests} 个稀疏网络 (α=0.3, β=0.1)...")
    
    for i in range(num_tests):
        G = generate_waxman_topology(
            num_nodes=20,
            alpha=0.3,
            beta=0.1,
            seed=i
        )
        
        if not nx.is_connected(G):
            all_connected = False
            print(f"  网络 {i}: 不连通！")
        else:
            print(f"  网络 {i}: 连通 (边数={G.number_of_edges()})")
    
    print(f"\n所有生成的网络都是连通的: {all_connected}")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Waxman 拓扑生成器测试套件")
    print("=" * 60 + "\n")
    
    test_basic_generation()
    test_parameter_variations()
    test_network_sizes()
    test_node_attributes()
    test_reproducibility()
    test_connectivity()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

