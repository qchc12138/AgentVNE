#!/usr/bin/env python3
"""
对概率矩阵文件进行 softmax 归一化

功能：
- 读取概率矩阵文件
- 对每行（每个VN节点）的概率值做softmax归一化
- 保存为新文件
"""

import os
import numpy as np

def softmax(x):
    """
    对输入数组进行softmax归一化
    
    Args:
        x: 输入数组
    
    Returns:
        softmax归一化后的数组
    """
    # 减去最大值以提高数值稳定性
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def load_probability_matrix(file_path: str):
    """
    从文件加载概率矩阵
    
    Args:
        file_path: 文件路径
    
    Returns:
        (prob_matrix, vn_node_names, sn_node_names, header_lines)
        - prob_matrix: 概率矩阵 numpy 数组 [N_v, N_s]
        - vn_node_names: VN节点名称列表
        - sn_node_names: SN节点名称列表
        - header_lines: 文件头部信息（前3行）
    """
    prob_matrix = []
    vn_node_names = []
    sn_node_names = []
    header_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 保存头部信息（前3行）
    header_lines = lines[:3]
    
    # 找到表头行（包含"VN节点\SN节点"的行）
    header_line_idx = None
    for i, line in enumerate(lines):
        if 'VN节点\\SN节点' in line or 'VN节点/SN节点' in line:
            header_line_idx = i
            break
    
    if header_line_idx is None:
        raise ValueError("未找到表头行")
    
    # 解析表头，获取SN节点名称
    header_line = lines[header_line_idx].strip()
    sn_node_names = [name.strip() for name in header_line.split(',')[1:]]  # 跳过第一个"VN节点\SN节点"
    
    # 解析数据行
    for i in range(header_line_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('='):
            break  # 遇到空行或分隔线，停止读取
        
        parts = line.split(',')
        if len(parts) < 2:
            continue
        
        vn_node_name = parts[0].strip()
        prob_values = [float(p.strip()) for p in parts[1:]]
        
        vn_node_names.append(vn_node_name)
        prob_matrix.append(prob_values)
    
    prob_matrix = np.array(prob_matrix)
    
    return prob_matrix, vn_node_names, sn_node_names, header_lines

def save_probability_matrix(prob_matrix, vn_node_names, sn_node_names, header_lines, output_path):
    """
    保存概率矩阵到文件
    
    Args:
        prob_matrix: 概率矩阵 numpy 数组 [N_v, N_s]
        vn_node_names: VN节点名称列表
        sn_node_names: SN节点名称列表
        header_lines: 文件头部信息
        output_path: 输出文件路径
    """
    N_v, N_s = prob_matrix.shape
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入头部信息（修改标题）
        for line in header_lines:
            if '【预训练模型】' in line:
                line = line.replace('【预训练模型】', '【预训练模型 - Softmax归一化】')
            elif '【微调模型】' in line:
                line = line.replace('【微调模型】', '【微调模型 - Softmax归一化】')
            f.write(line)
        
        # 写入空行
        f.write('\n')
        
        # 写入表头
        f.write('VN节点\\SN节点')
        for sn_name in sn_node_names:
            f.write(f',{sn_name}')
        f.write('\n')
        
        # 写入数据行
        for i in range(N_v):
            f.write(f'{vn_node_names[i]}')
            for j in range(N_s):
                f.write(f',{prob_matrix[i, j]:.6f}')
            f.write('\n')
        
        # 写入分隔线
        f.write('\n')
        f.write('=' * 80 + '\n')
    
    print(f"✓ 概率矩阵已保存到: {output_path}")

def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入文件路径
    input_file = os.path.join(script_dir, 'pretrain_prob_matrix.txt')
    
    # 输出文件路径
    output_file = os.path.join(script_dir, 'pretrain_prob_matrix_softmax.txt')
    
    print("=" * 80)
    print("概率矩阵 Softmax 归一化脚本")
    print("=" * 80)
    print(f"\n输入文件: {input_file}")
    print(f"输出文件: {output_file}\n")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 加载概率矩阵
    print("【步骤1】加载概率矩阵...")
    prob_matrix, vn_node_names, sn_node_names, header_lines = load_probability_matrix(input_file)
    print(f"  ✓ 加载成功")
    print(f"  VN节点数: {prob_matrix.shape[0]}")
    print(f"  SN节点数: {prob_matrix.shape[1]}")
    
    # 打印原始概率矩阵的统计信息
    print(f"\n【步骤2】原始概率矩阵统计:")
    for i, vn_name in enumerate(vn_node_names):
        row_sum = prob_matrix[i].sum()
        row_max = prob_matrix[i].max()
        row_min = prob_matrix[i].min()
        print(f"  {vn_name}: 和={row_sum:.6f}, 最大值={row_max:.6f}, 最小值={row_min:.6f}")
    
    # 对每行进行softmax归一化
    print(f"\n【步骤3】对每行进行softmax归一化...")
    softmax_matrix = np.zeros_like(prob_matrix)
    for i in range(prob_matrix.shape[0]):
        softmax_matrix[i] = softmax(prob_matrix[i])
    
    # 打印归一化后的统计信息
    print(f"\n【步骤4】归一化后概率矩阵统计:")
    for i, vn_name in enumerate(vn_node_names):
        row_sum = softmax_matrix[i].sum()
        row_max = softmax_matrix[i].max()
        row_min = softmax_matrix[i].min()
        print(f"  {vn_name}: 和={row_sum:.6f}, 最大值={row_max:.6f}, 最小值={row_min:.6f}")
    
    # 保存结果
    print(f"\n【步骤5】保存结果...")
    save_probability_matrix(softmax_matrix, vn_node_names, sn_node_names, header_lines, output_file)
    
    print(f"\n{'=' * 80}")
    print("处理完成！")
    print(f"{'=' * 80}\n")

if __name__ == '__main__':
    main()

