#!/usr/bin/env python3
"""生成底层网络拓扑（Substrate Network）"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'waxman'))

from topo_create import generate_waxman_topology
from networkx.readwrite import json_graph
import json


def generate_SN_topology():
    """生成底层网络拓扑"""
    
    print("生成 Waxman 底层拓扑...")
    G = generate_waxman_topology(
        num_nodes=10,
        alpha=0.6,
        beta=0.2,
        seed=42
    )
    
    data = json_graph.node_link_data(G)
    
    print("添加节点资源特征...")
    for i, node in enumerate(data['nodes']):
        if i in [6, 8]:
            node['cpu'] = 4
            node['memory'] = 4
            node['disk'] = 6
        else:
            node['cpu'] = 2
            node['memory'] = 2
            node['disk'] = 4
        
        node['bandwidth'] = 10
        node['comm_bandwidth'] = 10
        node['node_type'] = 'edge'
    
    print("添加边带宽信息...")
    for link in data['links']:
        if 'weight' in link:
            link['bandwidth'] = 10
        else:
            link['bandwidth'] = 10
            link['weight'] = 1.0
    
    data['directed'] = False
    data['multigraph'] = False
    
    output_file = '/home/yc2/mrt/a/topo/SN_topology.json'
    print(f"\n保存底层拓扑到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✓ 底层拓扑文件已生成")
    
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


