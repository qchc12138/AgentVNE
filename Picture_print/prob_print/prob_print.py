#!/usr/bin/env python3
"""
概率矩阵热力图绘制脚本

功能：
- 从CSV文件读取预训练模型和微调模型的概率矩阵
- 使用matplotlib绘制热力图
- 保存图片

使用方法：
    python3 prob_print.py
"""

import csv
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use('Agg')  # 使用非交互式后端

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 字体大小设置（参考 plot_avg_rt_from_csv.py）
TITLE_FONTSIZE = 36       # 标题字体大小
LABEL_FONTSIZE = 34       # 坐标轴标签字体大小
LEGEND_FONTSIZE = 32      # 图例字体大小
TICK_FONTSIZE = 34        # 坐标轴刻度字体大小

# 输入文件路径（相对于脚本目录）
PRETRAIN_FILE = "pretrain_prob_matrix.txt"
FINETUNING_FILE = "finetuning_prob_matrix.txt"

# 输出文件路径
OUTPUT_DIR = None  # None表示使用脚本所在目录
OUTPUT_PRETRAIN = "pretrain_prob_matrix_heatmap.png"
OUTPUT_FINETUNING = "finetuning_prob_matrix_heatmap.png"

# 图表样式
FIG_SIZE = (16, 10)       # 图表大小（宽, 高）
DPI = 200                # 分辨率
SHOW_NUMBERS = True        # 是否在热力图中显示数字（True=显示，False=不显示）


def load_probability_matrix(file_path: str) -> tuple[np.ndarray, list, list]:
    """
    从CSV文件加载概率矩阵
    
    Args:
        file_path: CSV文件路径
    
    Returns:
        (prob_matrix, vn_node_names, sn_node_names)
        - prob_matrix: 概率矩阵 numpy数组 [N_v, N_s]
        - vn_node_names: VN节点名称列表
        - sn_node_names: SN节点名称列表
    """
    prob_matrix = []
    vn_node_names = []
    sn_node_names = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 找到CSV数据开始的行（包含"VN节点\SN节点"的行）
        data_start_idx = None
        for i, line in enumerate(lines):
            if 'VN节点\\SN节点' in line or 'VN节点,SN节点' in line:
                data_start_idx = i
                # 解析SN节点名称
                parts = line.strip().split(',')
                if len(parts) > 1:
                    sn_node_names = parts[1:]  # 跳过第一个"VN节点\SN节点"
                break
        
        if data_start_idx is None:
            raise ValueError(f"无法找到CSV数据开始行: {file_path}")
        
        # 读取数据行
        for i in range(data_start_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('='):
                break
            
            parts = line.split(',')
            if len(parts) < 2:
                continue
            
            vn_name = parts[0]
            vn_node_names.append(vn_name)
            
            # 解析概率值
            probs = [float(x) for x in parts[1:]]
            prob_matrix.append(probs)
    
    prob_matrix = np.array(prob_matrix)
    
    return prob_matrix, vn_node_names, sn_node_names


def extract_node_id(node_name: str) -> str:
    """
    从节点名称中提取ID
    
    Args:
        node_name: 节点名称（如"SN0", "VN1"等）
    
    Returns:
        节点ID字符串（如"0", "1"等）
    """
    # 移除所有非数字字符，只保留数字
    match = re.search(r'\d+', node_name)
    return match.group() if match else node_name


def plot_heatmap(prob_matrix: np.ndarray, 
                  vn_node_names: list, 
                  sn_node_names: list,
                  title: str,
                  output_path: str,
                  figsize: tuple = (16, 10),
                  dpi: int = 200,
                  show_numbers: bool = True):
    """
    绘制概率矩阵热力图
    纵轴：VN节点，横轴：SN节点
    
    Args:
        prob_matrix: 概率矩阵 [N_v, N_s]
        vn_node_names: VN节点名称列表
        sn_node_names: SN节点名称列表
        title: 图表标题
        output_path: 输出文件路径
        figsize: 图表大小
        dpi: 分辨率
        show_numbers: 是否在单元格中显示数字（True=显示，False=不显示）
    """
    # 定义浅绿色系的colormap
    # 颜色范围：浅黄(0) → 浅绿 → 深绿
    colors = ["#f9f9e0", "#c8e6c9", "#81c784", "#388e3c"]
    cmap = LinearSegmentedColormap.from_list("light_green", colors)
    
    # 固定颜色范围为0-0.3（统一所有热力图的颜色尺度）
    vmin = 0.0
    vmax = 0.3
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 创建热力图，使用aspect='equal'使刻度长度相等
    # 使用浅绿色系配色方案
    im = ax.imshow(prob_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
    
    # 提取节点ID（只显示数字）
    vn_node_ids = [extract_node_id(name) for name in vn_node_names]
    sn_node_ids = [extract_node_id(name) for name in sn_node_names]
    
    # 设置坐标轴刻度
    # 横轴：SN节点
    ax.set_xticks(np.arange(len(sn_node_ids)))
    ax.set_xticklabels(sn_node_ids, fontsize=TICK_FONTSIZE)
    
    # 纵轴：VN节点
    ax.set_yticks(np.arange(len(vn_node_ids)))
    ax.set_yticklabels(vn_node_ids, fontsize=TICK_FONTSIZE)
    
    # 关键：添加白色缝隙（网格线偏移0.5个坐标单位）
    # 设置网格线位置为偏移0.5的位置（0.5, 1.5, 2.5...）
    x_grid_positions = np.arange(len(sn_node_ids)) + 0.5
    y_grid_positions = np.arange(len(vn_node_ids)) + 0.5
    
    # 绘制垂直网格线（横轴方向）
    for x_pos in x_grid_positions:
        ax.axvline(x=x_pos, color="white", linestyle="-", linewidth=2)
    
    # 绘制水平网格线（纵轴方向）
    for y_pos in y_grid_positions:
        ax.axhline(y=y_pos, color="white", linestyle="-", linewidth=2)
    
    # 绘制边界线（最外层的网格线）
    ax.axvline(x=-0.5, color="white", linestyle="-", linewidth=2)
    ax.axvline(x=len(sn_node_ids) - 0.5, color="white", linestyle="-", linewidth=2)
    ax.axhline(y=-0.5, color="white", linestyle="-", linewidth=2)
    ax.axhline(y=len(vn_node_ids) - 0.5, color="white", linestyle="-", linewidth=2)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # 在每个单元格中添加数值标注（如果启用）
    if show_numbers:
        for i in range(len(vn_node_ids)):  # VN节点（纵轴）
            for j in range(len(sn_node_ids)):  # SN节点（横轴）
                value = prob_matrix[i, j]
                # 根据数值大小选择文字颜色
                text_color = 'white' if value > prob_matrix.max() * 0.5 else 'black'
                text = ax.text(j, i, f'{value:.3f}',
                              ha="center", va="center", color=text_color,
                              fontsize=20)  # 数值字体稍小一些
    
    # 设置标题和标签
    # ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=20)
    ax.set_xlabel('Substrade Nodes', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Workflow Nodes', fontsize=LABEL_FONTSIZE, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label('概率值', fontsize=LABEL_FONTSIZE, fontweight='bold')
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ 热力图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定输出目录
    if OUTPUT_DIR is None:
        output_dir = script_dir
    else:
        output_dir = OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else os.path.join(script_dir, OUTPUT_DIR)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件路径
    pretrain_path = os.path.join(script_dir, PRETRAIN_FILE)
    finetuning_path = os.path.join(script_dir, FINETUNING_FILE)
    
    print("="*80)
    print("概率矩阵热力图绘制脚本")
    print("="*80)
    
    # 加载预训练模型的概率矩阵
    print(f"\n【加载预训练模型概率矩阵】")
    print(f"  文件: {pretrain_path}")
    pretrain_matrix, pretrain_vn_names, pretrain_sn_names = load_probability_matrix(pretrain_path)
    print(f"  ✓ 加载成功: {pretrain_matrix.shape[0]} 个VN节点, {pretrain_matrix.shape[1]} 个SN节点")
    
    # 加载微调模型的概率矩阵
    print(f"\n【加载微调模型概率矩阵】")
    print(f"  文件: {finetuning_path}")
    finetuning_matrix, finetuning_vn_names, finetuning_sn_names = load_probability_matrix(finetuning_path)
    print(f"  ✓ 加载成功: {finetuning_matrix.shape[0]} 个VN节点, {finetuning_matrix.shape[1]} 个SN节点")
    
    # 绘制预训练模型热力图
    print(f"\n【绘制预训练模型热力图】")
    print(f"  显示数字: {SHOW_NUMBERS}")
    pretrain_output_path = os.path.join(output_dir, OUTPUT_PRETRAIN)
    plot_heatmap(
        pretrain_matrix,
        pretrain_vn_names,
        pretrain_sn_names,
        "预训练模型概率矩阵",
        pretrain_output_path,
        figsize=FIG_SIZE,
        dpi=DPI,
        show_numbers=SHOW_NUMBERS
    )
    
    # 绘制微调模型热力图
    print(f"\n【绘制微调模型热力图】")
    print(f"  显示数字: {SHOW_NUMBERS}")
    finetuning_output_path = os.path.join(output_dir, OUTPUT_FINETUNING)
    plot_heatmap(
        finetuning_matrix,
        finetuning_vn_names,
        finetuning_sn_names,
        "微调模型概率矩阵",
        finetuning_output_path,
        figsize=FIG_SIZE,
        dpi=DPI,
        show_numbers=SHOW_NUMBERS
    )
    
    print(f"\n{'='*80}")
    print(f"所有热力图已保存到: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

