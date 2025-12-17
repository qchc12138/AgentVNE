"""
可视化 SN 拓扑文件
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

def load_topology(json_path):
    """加载拓扑 JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def visualize_topology(json_path, output_path=None, figsize=(14, 10)):
    """
    可视化 SN 拓扑
    
    Args:
        json_path: 拓扑 JSON 文件路径
        output_path: 输出图片路径（如果为None，则显示）
        figsize: 图片大小
    """
    # 加载拓扑数据
    data = load_topology(json_path)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建 NetworkX 图
    if data.get('directed', False):
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # 添加节点
    node_positions = {}
    node_attributes = {}
    for node in data['nodes']:
        node_id = node['id']
        G.add_node(node_id)
        
        # 获取位置（优先使用 pos，否则使用 x, y）
        if 'pos' in node and len(node['pos']) >= 2:
            pos = (node['pos'][0], node['pos'][1])
        elif 'x' in node and 'y' in node:
            pos = (node['x'], node['y'])
        else:
            pos = (0, 0)
        
        node_positions[node_id] = pos
        node_attributes[node_id] = {
            'cpu': node.get('cpu', 0),
            'memory': node.get('memory', 0),
            'disk': node.get('disk', 0),
            'bandwidth': node.get('bandwidth', 0),
            'node_type': node.get('node_type', 'unknown'),
            'bias_cpu': node.get('bias_cpu', 0.0),
        }
    
    # 添加边
    edge_attributes = {}
    for link in data['links']:
        source = link['source']
        target = link['target']
        G.add_edge(source, target)
        edge_attributes[(source, target)] = {
            'weight': link.get('weight', 1.0),
            'bandwidth': link.get('bandwidth', 0),
        }
    
    # 绘制边
    edges = G.edges()
    nx.draw_networkx_edges(G, node_positions, edges, 
                           edge_color='gray', 
                           width=1.5,
                           alpha=0.6,
                           ax=ax)
    
    # 根据资源大小设置节点大小和颜色
    cpu_values = [node_attributes[n].get('cpu', 0) for n in G.nodes()]
    max_cpu = max(cpu_values) if cpu_values else 1
    
    # 节点大小基于 CPU 资源
    node_sizes = [300 + node_attributes[n].get('cpu', 0) * 200 for n in G.nodes()]
    
    # 节点颜色基于 CPU 资源（使用 colormap）
    node_colors = [node_attributes[n].get('cpu', 0) / max_cpu for n in G.nodes()]
    
    # 绘制节点
    nodes = nx.draw_networkx_nodes(G, node_positions,
                                   node_color=node_colors,
                                   node_size=node_sizes,
                                   cmap=plt.cm.viridis,
                                   alpha=0.8,
                                   ax=ax)
    
    # 添加节点标签（显示节点 ID）
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, node_positions, labels,
                           font_size=10,
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    # 添加资源信息标注（在节点旁边）
    for node_id, pos in node_positions.items():
        attrs = node_attributes[node_id]
        # 创建资源信息文本
        info_text = f"CPU:{attrs['cpu']}\nMEM:{attrs['memory']}\nDISK:{attrs['disk']}"
        if attrs['bias_cpu'] > 0:
            info_text += f"\nBias:{attrs['bias_cpu']:.1f}"
        
        # 在节点右侧添加文本
        ax.text(pos[0] + 3, pos[1], info_text,
               fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
               verticalalignment='center',
               horizontalalignment='left')
    
    # 添加边权重标注（可选，如果边不太多的话）
    if len(edges) <= 20:  # 只在边不太多时显示权重
        edge_labels = {(e[0], e[1]): f"{edge_attributes.get((e[0], e[1]), {}).get('weight', 0):.1f}" 
                      for e in edges}
        nx.draw_networkx_edge_labels(G, node_positions, edge_labels,
                                    font_size=6,
                                    alpha=0.7,
                                    ax=ax)
    
    # 设置标题
    topology_name = os.path.basename(json_path).replace('.json', '')
    ax.set_title(f'SN Topology Visualization: {topology_name}\n'
                f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}',
                fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color=plt.cm.viridis(0.0), label='Low CPU'),
        mpatches.Patch(color=plt.cm.viridis(1.0), label='High CPU'),
        Line2D([0], [0], color='gray', linewidth=1.5, label='Link'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 移除坐标轴
    ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"拓扑可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 打印统计信息
    print(f"\n拓扑统计信息:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    cpu_total = sum(node_attributes[n].get('cpu', 0) for n in G.nodes())
    mem_total = sum(node_attributes[n].get('memory', 0) for n in G.nodes())
    disk_total = sum(node_attributes[n].get('disk', 0) for n in G.nodes())
    print(f"  总 CPU: {cpu_total}")
    print(f"  总 Memory: {mem_total}")
    print(f"  总 Disk: {disk_total}")
    
    # 打印节点详细信息
    print(f"\n节点详细信息:")
    for node_id in sorted(G.nodes()):
        attrs = node_attributes[node_id]
        print(f"  节点 {node_id}: CPU={attrs['cpu']}, MEM={attrs['memory']}, "
              f"DISK={attrs['disk']}, Type={attrs['node_type']}, "
              f"Bias_CPU={attrs['bias_cpu']}")

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 默认拓扑文件路径
    topology_file = os.path.join(script_dir, 'SN_topology_2.json')
    
    # 输出图片路径
    output_file = os.path.join(script_dir, 'SN_topology_2_visualization.png')
    
    # 检查文件是否存在
    if not os.path.exists(topology_file):
        print(f"错误: 拓扑文件不存在: {topology_file}")
        exit(1)
    
    # 可视化拓扑
    print(f"正在可视化拓扑文件: {topology_file}")
    visualize_topology(topology_file, output_path=output_file)
    print(f"\n可视化完成！")

