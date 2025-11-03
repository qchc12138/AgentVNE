#!/usr/bin/env python3
"""
快速入门：30秒生成 Waxman 拓扑
"""

from topo_create import generate_waxman_topology, print_topology_info
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import json
import os

# 生成拓扑（使用论文标准参数 α=0.5, β=0.2）
G = generate_waxman_topology(
    num_nodes=10,
    alpha=0.6, # 全局连接系数，控制连接的总数
    beta=0.2,
    seed=42
)

# 打印信息
print("生成的 Waxman 拓扑:")
print_topology_info(G)

# 查看节点位置
print("\n前3个节点的位置:")
for i in range(10):
    print(f"  节点{i}: x={G.nodes[i]['x']:.2f}, y={G.nodes[i]['y']:.2f}")

# 保存为 JSON 文件
print("\n正在保存拓扑为 JSON 格式...")
try:
    data = json_graph.node_link_data(G)
    json_filename = 'topology.json'
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ 拓扑已保存为 JSON 格式: {json_filename}")
except Exception as e:
    print(f"✗ 保存 JSON 文件时出错: {e}")

# 绘制并保存为 PNG 文件
print("\n正在绘制拓扑图...")
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
    
    ax.set_title(f'Waxman 拓扑 (α={0.6}, β={0.2}, n={10})')
    ax.axis('off')
    plt.tight_layout()
    
    # 保存图像
    png_filename = 'waxman_topology.png'
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 拓扑图已保存为 PNG 格式: {png_filename}")
except Exception as e:
    print(f"✗ 保存 PNG 文件时出错: {e}")
    print("注意: matplotlib 可能需要安装，运行 'pip install matplotlib'")

print("\n完成！")

