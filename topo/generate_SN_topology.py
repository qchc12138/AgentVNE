#!/usr/bin/env python3
"""
生成底层网络拓扑（Substrate Network）
使用 Waxman 模型生成拓扑，并添加节点资源特征
"""

import sys
import os

# 添加 waxman 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'waxman'))

from topo_create import generate_waxman_topology
from networkx.readwrite import json_graph
import json


def generate_SN_topology():
    """生成底层网络拓扑"""
    
    # 使用与 quick_start.py 相同的配置
    print("生成 Waxman 底层拓扑...")
    G = generate_waxman_topology(
        num_nodes=10,
        alpha=0.6,  # 全局连接系数
        beta=0.2,
        seed=42
    )
    
    # 转换为 JSON 格式
    data = json_graph.node_link_data(G)
    
    # 为每个节点添加资源特征
    print("添加节点资源特征...")
    for i, node in enumerate(data['nodes']):
        # 默认节点配置
        if i in [6, 8]:
            # id 为 6 和 8 的节点
            node['cpu'] = 4
            node['memory'] = 4
            node['disk'] = 6
        else:
            # 其他节点
            node['cpu'] = 2
            node['memory'] = 2
            node['disk'] = 4
        
        # 所有节点共有的配置
        node['bandwidth'] = 10
        node['comm_bandwidth'] = 10
        node['node_type'] = 'edge'
    
    # 为每个边添加带宽信息
    print("添加边带宽信息...")
    for link in data['links']:
        # 使用 weight 作为距离，添加 bandwidth
        if 'weight' in link:
            # 将距离转换为带宽，距离越大带宽越小
            # 这里简化为固定带宽
            link['bandwidth'] = 10
        else:
            link['bandwidth'] = 10
            link['weight'] = 1.0
    
    # 确保 directed 和 multigraph 字段
    data['directed'] = False
    data['multigraph'] = False
    
    # 保存文件
    output_file = '/home/yc2/mrt/a/topo/SN_topology.json'
    print(f"\n保存底层拓扑到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✓ 底层拓扑文件已生成")
    
    # 打印统计信息
    print(f"\n拓扑统计:")
    print(f"  节点数: {len(data['nodes'])}")
    print(f"  边数: {len(data['links'])}")
    
    print(f"\n节点资源配置:")
    for node in data['nodes']:
        print(f"  节点 {node['id']}: CPU={node['cpu']}, Memory={node['memory']}, Disk={node['disk']}")
    
    return output_file


if __name__ == '__main__':
    generate_SN_topology()
    print("\n完成！")


