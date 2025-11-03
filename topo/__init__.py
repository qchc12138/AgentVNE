"""
拓扑生成模块

提供两种拓扑生成器：
1. Waxman拓扑 - 随机网络拓扑生成
2. 星型拓扑 - 自定义资源的星型网络
"""

__all__ = [
    # Waxman拓扑
    'generate_waxman_topology',
    'generate_topology',
    'print_topology_info',
    # 星型拓扑
    'create_star_topology',
    'print_star_topology_info',
    'get_node_resources',
    'get_edge_bandwidth',
    'export_star_topology',
    'visualize_star_topology',
]

