#!/usr/bin/env python3
"""
可视化 Workflow 拓扑文件

功能：
- 读取 workflow JSON 文件
- 可视化节点和边的连接关系
- 显示节点的资源需求和约束信息
- 保存为图片
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

# 配置中文字体支持
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_workflow(json_path):
    """加载 workflow JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def visualize_workflow(json_path, output_path=None, figsize=(14, 10)):
    """
    可视化 Workflow 拓扑
    
    Args:
        json_path: workflow JSON 文件路径
        output_path: 输出图片路径（如果为None，则显示）
        figsize: 图片大小
    """
    # 加载数据
    data = load_workflow(json_path)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建 NetworkX 图
    if data.get('directed', False):
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # 添加节点
    node_attributes = {}
    constraint_nodes = []
    
    for node in data['nodes']:
        node_id = node['id']
        G.add_node(node_id)
        
        node_attributes[node_id] = {
            'cpu': node.get('cpu', 0),
            'memory': node.get('memory', 0),
            'disk': node.get('disk', 0),
            'constraint_node': node.get('constraint_node'),
        }
        
        # 记录约束节点
        if node.get('constraint_node') is not None:
            constraint_nodes.append(node_id)
    
    # 添加边
    for link in data['links']:
        source = link['source']
        target = link['target']
        G.add_edge(source, target)
    
    # 使用 spring 布局（适合有向图）
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 计算节点大小（基于资源需求总和）
    node_sizes = []
    node_colors = []
    for node_id in G.nodes():
        attrs = node_attributes[node_id]
        total_resource = attrs['cpu'] + attrs['memory'] + attrs['disk']
        node_sizes.append(500 + total_resource * 300)
        
        # 节点颜色：约束节点用红色，其他用蓝色系（基于CPU资源）
        if node_id in constraint_nodes:
            node_colors.append('red')
        else:
            # 使用CPU资源值映射到颜色（0-1范围）
            cpu_value = attrs['cpu']
            node_colors.append(plt.cm.Blues(0.3 + cpu_value * 0.7))
    
    # 绘制边
    if data.get('directed', False):
        # 有向图：使用箭头
        nx.draw_networkx_edges(G, pos,
                               edge_color='gray',
                               width=2,
                               alpha=0.6,
                               arrows=True,
                               arrowsize=20,
                               arrowstyle='->',
                               ax=ax)
    else:
        # 无向图：不使用箭头
        nx.draw_networkx_edges(G, pos,
                               edge_color='gray',
                               width=2,
                               alpha=0.6,
                               ax=ax)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.8,
                           ax=ax)
    
    # 添加节点标签（显示节点 ID）
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=12,
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    # 添加资源信息标注（在节点旁边）
    for node_id, position in pos.items():
        attrs = node_attributes[node_id]
        # 创建资源信息文本
        info_text = f"CPU:{attrs['cpu']:.2f}\nMEM:{attrs['memory']:.2f}\nDISK:{attrs['disk']:.2f}"
        if attrs['constraint_node'] is not None:
            info_text += f"\n约束→SN{attrs['constraint_node']}"
        
        # 在节点右侧添加文本
        ax.text(position[0] + 0.15, position[1], info_text,
               fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
               verticalalignment='center',
               horizontalalignment='left')
    
    # 设置标题
    workflow_name = os.path.basename(json_path).replace('.json', '')
    ax.set_title(f'Workflow Topology Visualization: {workflow_name}\n'
                f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, '
                f'Directed: {data.get("directed", False)}',
                fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='red', label='约束节点'),
        mpatches.Patch(color=plt.cm.Blues(0.5), label='普通节点'),
        Line2D([0], [0], color='gray', linewidth=2, label='边'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 移除坐标轴
    ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Workflow可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 打印统计信息
    print(f"\nWorkflow统计信息:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  有向图: {data.get('directed', False)}")
    print(f"  约束节点数: {len(constraint_nodes)}")
    if constraint_nodes:
        print(f"  约束节点: {constraint_nodes}")
    
    cpu_total = sum(node_attributes[n].get('cpu', 0) for n in G.nodes())
    mem_total = sum(node_attributes[n].get('memory', 0) for n in G.nodes())
    disk_total = sum(node_attributes[n].get('disk', 0) for n in G.nodes())
    print(f"  总 CPU 需求: {cpu_total:.2f}")
    print(f"  总 Memory 需求: {mem_total:.2f}")
    print(f"  总 Disk 需求: {disk_total:.2f}")
    
    # 打印节点详细信息
    print(f"\n节点详细信息:")
    for node_id in sorted(G.nodes()):
        attrs = node_attributes[node_id]
        constraint_info = f", 约束→SN{attrs['constraint_node']}" if attrs['constraint_node'] is not None else ""
        print(f"  节点 {node_id}: CPU={attrs['cpu']:.2f}, MEM={attrs['memory']:.2f}, "
              f"DISK={attrs['disk']:.2f}{constraint_info}")
    
    # 打印边的信息
    print(f"\n边信息:")
    for link in data.get('links', []):
        source = link['source']
        target = link['target']
        weight = link.get('weight', 1.0)
        bandwidth = link.get('bandwidth', 0.0)
        print(f"  {source} → {target}: weight={weight:.2f}, bandwidth={bandwidth:.2f}")

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 默认 workflow 文件路径
    workflow_file = os.path.join(script_dir, 'workflow1_topo.json')
    
    # 输出图片路径
    output_file = os.path.join(script_dir, 'workflow1_topo_visualization.pdf')
    
    # 检查文件是否存在
    if not os.path.exists(workflow_file):
        print(f"错误: Workflow文件不存在: {workflow_file}")
        exit(1)
    
    # 可视化 workflow
    print(f"正在可视化 Workflow 文件: {workflow_file}")
    visualize_workflow(workflow_file, output_path=output_file)
    print(f"\n可视化完成！")

