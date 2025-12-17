"""
概率矩阵测试脚本：加载两个模型，计算并比较概率矩阵
"""
import json
import os
from datetime import datetime
from typing import Dict, Optional

import torch
import numpy as np
from torch_geometric.data import Data

from model_1 import SimuVNE
from env import SimuVNEEnv, WorkflowGenerator


def load_model(model_path: str, device: str = 'cpu') -> SimuVNE:
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
    
    Returns:
        加载好的策略网络
    """
    policy = SimuVNE()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        policy.load_state_dict(state_dict, strict=False)
        print(f"  ✓ 成功加载模型: {model_path}")
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")
    
    policy.to(device)
    policy.eval()  # 设置为评估模式
    return policy


def apply_bias_to_sn_features(sn: Data, env: SimuVNEEnv) -> Data:
    """
    在策略网络前向传播前，临时修改SN特征的CPU维度为 (cpu_res + bias_cpu) 归一化版本。
    不修改原始sn对象，返回新的Data对象。
    
    Args:
        sn: SN状态Data对象
        env: 环境对象，用于获取bias_cpu和最大容量
    
    Returns:
        新的Data对象，CPU特征已加入bias
    """
    # 创建新的Data对象，避免修改原始sn
    sn_with_bias = Data(x=sn.x.clone(), edge_index=sn.edge_index.clone())
    
    cpu_max = float(env._sn_max_capacity.get('cpu_max', 1.0)) + 1e-8
    
    # 获取SN节点ID列表（与env.get_sn_state()中的顺序一致）
    sn_node_list = sorted(env.G_sn.nodes())
    
    # 修改每个节点的CPU特征
    for idx in range(sn_with_bias.x.size(0)):
        if idx < len(sn_node_list):
            sn_node_id = sn_node_list[idx]
            node = env.G_sn.nodes[sn_node_id]
            
            # 获取当前剩余CPU和固定bias_cpu
            cpu_res = float(node.get('cpu_res', node.get('cpu', 0.0)))
            bias_cpu = float(node.get('bias_cpu', 0.0))
            
            # 计算加入bias后的归一化CPU特征
            cpu_norm_with_bias = (cpu_res + bias_cpu) / cpu_max
            
            # 只修改CPU特征（第0维），其他特征保持不变
            sn_with_bias.x[idx, 0] = cpu_norm_with_bias
    
    return sn_with_bias


def compute_probability_matrix(policy: SimuVNE, vn: Data, sn: Data, device: str = 'cpu') -> np.ndarray:
    """
    计算概率矩阵
    
    Args:
        policy: 策略网络
        vn: VN图数据
        sn: SN图数据（已应用bias）
        device: 设备
    
    Returns:
        概率矩阵 numpy 数组 [N_v, N_s]
    """
    vn = vn.to(device)
    sn = sn.to(device)
    
    with torch.no_grad():
        probs_matrix = policy(vn, sn)  # [N_v, N_s]
    
    return probs_matrix.cpu().numpy()


def print_probability_matrix(probs_matrix: np.ndarray, model_name: str, vn_node_names: list = None, sn_node_names: list = None):
    """
    打印概率矩阵
    
    Args:
        probs_matrix: 概率矩阵 [N_v, N_s]
        model_name: 模型名称
        vn_node_names: VN节点名称列表（可选）
        sn_node_names: SN节点名称列表（可选）
    """
    N_v, N_s = probs_matrix.shape
    
    print(f"\n{'='*80}")
    print(f"【{model_name}】概率矩阵 [VN节点数={N_v}, SN节点数={N_s}]")
    print(f"{'='*80}")
    
    # 打印表头
    if sn_node_names:
        header = "VN节点\\SN节点"
        for sn_name in sn_node_names:
            header += f"\t{sn_name:>8}"
        print(header)
    else:
        header = "VN节点\\SN节点"
        for j in range(N_s):
            header += f"\tSN{j:>3}"
        print(header)
    
    # 打印每一行
    for i in range(N_v):
        vn_name = vn_node_names[i] if vn_node_names and i < len(vn_node_names) else f"VN{i}"
        row_str = f"{vn_name:>10}"
        for j in range(N_s):
            prob = probs_matrix[i, j]
            row_str += f"\t{prob:>8.4f}"
        print(row_str)
    
    print(f"{'='*80}\n")


def save_probability_matrix(probs_matrix: np.ndarray, model_name: str, output_path: str, 
                           vn_node_names: list = None, sn_node_names: list = None):
    """
    保存概率矩阵到文件
    
    Args:
        probs_matrix: 概率矩阵 [N_v, N_s]
        model_name: 模型名称
        output_path: 输出文件路径
        vn_node_names: VN节点名称列表（可选）
        sn_node_names: SN节点名称列表（可选）
    """
    N_v, N_s = probs_matrix.shape
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"【{model_name}】概率矩阵 [VN节点数={N_v}, SN节点数={N_s}]\n")
        f.write(f"{'='*80}\n\n")
        
        # 保存为CSV格式
        f.write("VN节点\\SN节点")
        if sn_node_names:
            for sn_name in sn_node_names:
                f.write(f",{sn_name}")
        else:
            for j in range(N_s):
                f.write(f",SN{j}")
        f.write("\n")
        
        for i in range(N_v):
            vn_name = vn_node_names[i] if vn_node_names and i < len(vn_node_names) else f"VN{i}"
            f.write(f"{vn_name}")
            for j in range(N_s):
                prob = probs_matrix[i, j]
                f.write(f",{prob:.6f}")
            f.write("\n")
        
        f.write(f"\n{'='*80}\n")
    
    print(f"  ✓ 概率矩阵已保存到: {output_path}")


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型路径
    pretrain_model_path = "/home/zrz/AgentVNE/AgentVNE/pretrain_outputs/checkpoint_latest.pt"
    finetuning_model_path = "/home/zrz/AgentVNE/AgentVNE/finetuning_output/policy_network_latest.pth"
    
    # SN和Workflow路径
    sn_topology_path = os.path.join(script_dir, 'topo', 'SN_topology_2.json')
    workflow_path = os.path.join(script_dir, 'workflow_topo', 'workflow1_topo.json')
    
    device = 'cpu'
    
    print("="*80)
    print("概率矩阵测试脚本")
    print("="*80)
    
    # 1. 加载模型
    print("\n【步骤1】加载模型...")
    print(f"  预训练模型: {pretrain_model_path}")
    print(f"  微调模型: {finetuning_model_path}")
    
    pretrain_policy = load_model(pretrain_model_path, device=device)
    finetuning_policy = load_model(finetuning_model_path, device=device)
    
    # 2. 创建环境并加载SN和Workflow
    print("\n【步骤2】加载SN拓扑和Workflow...")
    print(f"  SN拓扑: {sn_topology_path}")
    print(f"  Workflow: {workflow_path}")
    
    env = SimuVNEEnv(
        sn_topology_path=sn_topology_path,
        device=device,
        penalty=-150.0,
        max_arrived_tasks=1
    )
    env.reset()
    
    # 获取SN容量用于VN特征归一化
    sn_capacity = env.get_sn_max_capacity()
    
    # 创建WorkflowGenerator并加载workflow
    workflow_types = {
        'workflow1': workflow_path
    }
    wf_gen = WorkflowGenerator(
        workflow_types=workflow_types,
        arrival_rate=0.05,
        mean_lifetime=10.0,
        seed=42,
        sn_capacity_for_norm=sn_capacity
    )
    
    # 加载workflow
    vn = wf_gen.load_workflow_graph('workflow1')
    print(f"  ✓ Workflow加载成功: {vn.x.size(0)} 个VN节点")
    
    # 获取SN状态
    sn_state = env.get_sn_state()
    print(f"  ✓ SN状态获取成功: {sn_state.x.size(0)} 个SN节点")
    
    # 应用bias到SN特征
    sn_with_bias = apply_bias_to_sn_features(sn_state, env)
    print(f"  ✓ SN特征bias应用完成")
    
    # 3. 计算概率矩阵
    print("\n【步骤3】计算概率矩阵...")
    
    # 预训练模型
    print("  计算预训练模型的概率矩阵...")
    pretrain_probs = compute_probability_matrix(pretrain_policy, vn, sn_with_bias, device=device)
    
    # 微调模型
    print("  计算微调模型的概率矩阵...")
    finetuning_probs = compute_probability_matrix(finetuning_policy, vn, sn_with_bias, device=device)
    
    # 4. 打印结果
    print("\n【步骤4】打印概率矩阵...")
    
    # 获取节点名称（可选）
    sn_node_list = sorted(env.G_sn.nodes())
    sn_node_names = [f"SN{sn_id}" for sn_id in sn_node_list]
    vn_node_names = [f"VN{i}" for i in range(vn.x.size(0))]
    
    print_probability_matrix(pretrain_probs, "预训练模型", vn_node_names, sn_node_names)
    print_probability_matrix(finetuning_probs, "微调模型", vn_node_names, sn_node_names)
    
    # 5. 保存结果
    print("\n【步骤5】保存概率矩阵到文件...")
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(script_dir, 'prob_test_output')
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 创建带时间戳的子文件夹
    run_dir = os.path.join(base_output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"  创建输出文件夹: {run_dir}")
    
    # 保存预训练模型的概率矩阵
    pretrain_output_path = os.path.join(run_dir, 'pretrain_prob_matrix.txt')
    save_probability_matrix(pretrain_probs, "预训练模型", pretrain_output_path, vn_node_names, sn_node_names)
    
    # 保存微调模型的概率矩阵
    finetuning_output_path = os.path.join(run_dir, 'finetuning_prob_matrix.txt')
    save_probability_matrix(finetuning_probs, "微调模型", finetuning_output_path, vn_node_names, sn_node_names)
    
    # 保存对比结果（两个矩阵的差值）
    diff_probs = finetuning_probs - pretrain_probs
    diff_output_path = os.path.join(run_dir, 'prob_matrix_diff.txt')
    save_probability_matrix(diff_probs, "概率矩阵差值（微调-预训练）", diff_output_path, vn_node_names, sn_node_names)
    
    # 保存为numpy格式（便于后续分析）
    np_output_path = os.path.join(run_dir, 'prob_matrices.npz')
    np.savez(np_output_path, 
             pretrain=pretrain_probs,
             finetuning=finetuning_probs,
             diff=diff_probs)
    print(f"  ✓ NumPy格式已保存到: {np_output_path}")
    
    # 6. 打印统计信息
    print("\n【步骤6】统计信息...")
    print(f"  预训练模型概率矩阵:")
    print(f"    形状: {pretrain_probs.shape}")
    print(f"    最小值: {pretrain_probs.min():.6f}")
    print(f"    最大值: {pretrain_probs.max():.6f}")
    print(f"    均值: {pretrain_probs.mean():.6f}")
    print(f"    标准差: {pretrain_probs.std():.6f}")
    
    print(f"\n  微调模型概率矩阵:")
    print(f"    形状: {finetuning_probs.shape}")
    print(f"    最小值: {finetuning_probs.min():.6f}")
    print(f"    最大值: {finetuning_probs.max():.6f}")
    print(f"    均值: {finetuning_probs.mean():.6f}")
    print(f"    标准差: {finetuning_probs.std():.6f}")
    
    print(f"\n  概率矩阵差值（微调-预训练）:")
    print(f"    最小值: {diff_probs.min():.6f}")
    print(f"    最大值: {diff_probs.max():.6f}")
    print(f"    均值: {diff_probs.mean():.6f}")
    print(f"    标准差: {diff_probs.std():.6f}")
    print(f"    绝对差值均值: {np.abs(diff_probs).mean():.6f}")
    
    print(f"\n{'='*80}")
    print(f"所有结果已保存到: {run_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

