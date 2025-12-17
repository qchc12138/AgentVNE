"""
概率矩阵测试脚本：处理多个workflow，使用微调模型进行放置，在处理最后一个workflow时打印两个模型的概率矩阵
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional

import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx

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


def generate_priority_lists(probs_matrix: torch.Tensor) -> List[List[int]]:
    """根据概率降序生成优先级列表（测试模式：不采样，直接按概率排序）"""
    N_v, N_s = probs_matrix.shape
    priority_lists = []
    
    for i in range(N_v):
        probs = probs_matrix[i]  # [N_s] 获取该VN节点对所有SN节点的概率
        # 按概率降序排序，获取索引
        _, sorted_indices = torch.sort(probs, descending=True)
        priority_list = sorted_indices.tolist()
        priority_lists.append(priority_list)
    
    return priority_lists


def get_vn_neighbors(vn: Data) -> Dict[int, Set[int]]:
    """获取VN节点的邻居关系（无向，即使边是有向的）"""
    neighbors = {i: set() for i in range(vn.x.size(0))}
    edge_index = vn.edge_index
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        neighbors[u].add(v)  # u的邻居包括v
        neighbors[v].add(u)  # v的邻居包括u（即使边是有向的）
    return neighbors


def get_sn_k_hop_neighbors(env: SimuVNEEnv, sn_node_id: int, k: int) -> Set[int]:
    """获取SN节点的k跳邻居（包括k跳内的所有节点）"""
    if k == 0:
        return {sn_node_id}
    # 使用networkx的single_source_shortest_path_length
    paths = nx.single_source_shortest_path_length(env.G_sn, sn_node_id, cutoff=k)
    return set(paths.keys())  # 包括距离0到k的所有节点


def check_and_deduct_resource(env: SimuVNEEnv, sn_node_id: int, vn_node_idx: int, vn: Data, verbose: bool = False) -> bool:
    """
    检查资源并立即扣减（如果资源足够）
    
    Args:
        env: 环境对象
        sn_node_id: SN节点ID
        vn_node_idx: VN节点索引
        vn: VN图数据
        verbose: 是否打印详细信息
    
    Returns:
        True if 资源足够且已扣减, False otherwise
    """
    sn_node = env.G_sn.nodes[sn_node_id]
    vn_feats = vn.x[vn_node_idx]
    
    # 计算绝对资源需求
    cpu_demand = float(vn_feats[0].item()) * (env._sn_max_capacity['cpu_max'] + 1e-8)
    mem_demand = float(vn_feats[1].item()) * (env._sn_max_capacity['mem_max'] + 1e-8)
    disk_demand = float(vn_feats[2].item()) * (env._sn_max_capacity['disk_max'] + 1e-8)
    
    # 检查资源是否足够
    if cpu_demand > sn_node['cpu_res'] + 1e-9:
        return False
    if mem_demand > sn_node['mem_res'] + 1e-9:
        return False
    if disk_demand > sn_node['disk_res'] + 1e-9:
        return False
    
    # 立即扣减资源
    sn_node['cpu_res'] -= cpu_demand
    sn_node['mem_res'] -= mem_demand
    sn_node['disk_res'] -= disk_demand
    
    return True


def rollback_resource_deductions(env: SimuVNEEnv, deduction_history: List[Tuple[int, int]], vn: Data, verbose: bool = False):
    """
    回滚资源扣减
    
    Args:
        env: 环境对象
        deduction_history: [(sn_node_id, vn_node_idx), ...] 资源扣减历史记录
        vn: VN图数据
        verbose: 是否打印详细信息
    """
    for sn_node_id, vn_node_idx in deduction_history:
        sn_node = env.G_sn.nodes[sn_node_id]
        vn_feats = vn.x[vn_node_idx]
        
        # 计算需要恢复的资源
        cpu_restore = float(vn_feats[0].item()) * (env._sn_max_capacity['cpu_max'] + 1e-8)
        mem_restore = float(vn_feats[1].item()) * (env._sn_max_capacity['mem_max'] + 1e-8)
        disk_restore = float(vn_feats[2].item()) * (env._sn_max_capacity['disk_max'] + 1e-8)
        
        # 恢复资源
        sn_node['cpu_res'] += cpu_restore
        sn_node['mem_res'] += mem_restore
        sn_node['disk_res'] += disk_restore


def place_workflow(policy: SimuVNE, vn: Data, sn_state: Data, env: SimuVNEEnv, device: str = 'cpu', k_hop: int = 1, verbose: bool = False) -> Tuple[Dict[int, int], bool, torch.Tensor]:
    """
    使用策略网络放置workflow（参考 fine_tuning_2_test.py 的放置逻辑）
    
    Args:
        policy: 策略网络
        vn: VN图数据
        sn_state: SN状态
        env: 环境对象
        device: 设备
        k_hop: k跳搜索参数
        verbose: 是否打印详细信息
    
    Returns:
        (mapping, success, probs_matrix): mapping是VN节点索引到SN节点ID的映射，success表示是否成功放置，probs_matrix是概率矩阵
    """
    vn = vn.to(device)
    sn_state = sn_state.to(device)
    
    # 应用bias到SN特征
    sn_with_bias = apply_bias_to_sn_features(sn_state, env)
    
    # 计算概率矩阵
    with torch.no_grad():
        probs_matrix = policy(vn, sn_with_bias)  # [N_v, N_s]
    
    N_v, N_s = probs_matrix.shape
    
    # 获取SN节点ID列表（用于索引映射）
    sn_node_list = sorted(env.G_sn.nodes())
    
    # 初始化映射和记录
    mapping: Dict[int, int] = {}  # VN节点索引 -> SN节点ID
    resource_deduction_history: List[Tuple[int, int]] = []  # (SN节点ID, VN节点索引)
    constraint_placed_vn: Set[int] = set()  # 已放置的约束节点集合
    
    # 约束节点优先放置
    constraint_nodes = getattr(vn, 'constraint_nodes', [None] * N_v)
    if len(constraint_nodes) != N_v:
        constraint_nodes = [None] * N_v
    
    for vn_idx in range(N_v):
        constraint_node_id = constraint_nodes[vn_idx] if vn_idx < len(constraint_nodes) else None
        if constraint_node_id is not None and constraint_node_id in env.G_sn.nodes():
            if check_and_deduct_resource(env, constraint_node_id, vn_idx, vn, verbose=verbose):
                mapping[vn_idx] = constraint_node_id
                constraint_placed_vn.add(vn_idx)
                resource_deduction_history.append((constraint_node_id, vn_idx))
                if verbose:
                    sn_node = env.G_sn.nodes[constraint_node_id]
                    print(f"  [放置] VN节点{vn_idx} → SN节点{constraint_node_id} (约束节点)")
                    print(f"    SN节点{constraint_node_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
            else:
                rollback_resource_deductions(env, resource_deduction_history, vn, verbose=verbose)
                return {}, False, probs_matrix
    
    if len(constraint_placed_vn) == N_v:
        return mapping, True, probs_matrix
    
    # 生成优先级列表（按概率降序排序，测试模式）
    priority_lists = generate_priority_lists(probs_matrix)
    
    # 打印采样得到的优先级列表（如果verbose=True）
    if verbose:
        print(f"\n【采样得到的优先级列表】:")
        for vn_idx in range(N_v):
            if vn_idx not in constraint_placed_vn:
                priority_sn_ids = [sn_node_list[idx] for idx in priority_lists[vn_idx]]
                priority_probs = [float(probs_matrix[vn_idx][idx].item()) for idx in priority_lists[vn_idx]]
                print(f"  VN节点{vn_idx}: SN节点优先级序列 = {priority_sn_ids}")
                print(f"    对应概率值 = {[f'{p:.4f}' for p in priority_probs]}")
    
    # VN资源需求与度
    vn_neighbors = get_vn_neighbors(vn)
    vn_degrees = {i: len(vn_neighbors[i]) for i in range(N_v)}
    vn_resource_demands = {i: float(vn.x[i][0] + vn.x[i][1] + vn.x[i][2]) for i in range(N_v)}
    
    # 选择首个非约束 VN（资源需求最大）
    non_constraint_vn = [i for i in range(N_v) if i not in constraint_placed_vn]
    if not non_constraint_vn:
        return mapping, True, probs_matrix
    first_vn = max(non_constraint_vn, key=lambda i: vn_resource_demands[i])
    
    placed_first = False
    for first_sn_idx in priority_lists[first_vn]:
        first_sn_id = sn_node_list[first_sn_idx]
        if check_and_deduct_resource(env, first_sn_id, first_vn, vn, verbose=verbose):
            mapping[first_vn] = first_sn_id
            resource_deduction_history.append((first_sn_id, first_vn))
            placed_first = True
            if verbose:
                sn_node = env.G_sn.nodes[first_sn_id]
                print(f"  [放置] VN节点{first_vn} → SN节点{first_sn_id} (首个非约束节点)")
                print(f"    SN节点{first_sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
            break
    if not placed_first:
        rollback_resource_deductions(env, resource_deduction_history, vn, verbose=verbose)
        return {}, False, probs_matrix
    
    placed_vn: Set[int] = constraint_placed_vn.copy()
    placed_vn.add(first_vn)
    queue = [first_vn]
    while queue and len(placed_vn) < N_v:
        new_placed: List[int] = []
        for vi in queue:
            vi_sn_id = mapping[vi]
            unplaced_neighbors = [u for u in vn_neighbors[vi] if u not in placed_vn and u not in constraint_placed_vn]
            
            for u in unplaced_neighbors:
                # 尝试同SN
                if check_and_deduct_resource(env, vi_sn_id, u, vn, verbose=verbose):
                    mapping[u] = vi_sn_id
                    placed_vn.add(u)
                    new_placed.append(u)
                    resource_deduction_history.append((vi_sn_id, u))
                    if verbose:
                        sn_node = env.G_sn.nodes[vi_sn_id]
                        print(f"  [放置] VN节点{u} → SN节点{vi_sn_id} (同SN节点)")
                        print(f"    SN节点{vi_sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                    continue
                
                # k-hop 搜索
                k = 1
                max_k = len(sn_node_list)
                placed = False
                while k <= max_k and not placed:
                    k_hop_neighbors = get_sn_k_hop_neighbors(env, vi_sn_id, k)
                    for sn_idx in priority_lists[u]:
                        sn_id = sn_node_list[sn_idx]
                        if sn_id in k_hop_neighbors and check_and_deduct_resource(env, sn_id, u, vn, verbose=verbose):
                            mapping[u] = sn_id
                            placed_vn.add(u)
                            new_placed.append(u)
                            resource_deduction_history.append((sn_id, u))
                            placed = True
                            if verbose:
                                sn_node = env.G_sn.nodes[sn_id]
                                print(f"  [放置] VN节点{u} → SN节点{sn_id} (k={k}跳邻居)")
                                print(f"    SN节点{sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                            break
                    k += 1
                if not placed:
                    rollback_resource_deductions(env, resource_deduction_history, vn, verbose=verbose)
                    return {}, False, probs_matrix
        
        queue = sorted(
            [i for i in new_placed if i not in constraint_placed_vn],
            key=lambda i: (vn_degrees[i], vn_resource_demands[i]),
            reverse=True
        )
    
    if len(placed_vn) < N_v:
        rollback_resource_deductions(env, resource_deduction_history, vn, verbose=verbose)
        return {}, False, probs_matrix
    
    return mapping, True, probs_matrix


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
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(script_dir, 'prob_test_output')
    os.makedirs(base_output_dir, exist_ok=True)
    run_dir = os.path.join(base_output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"创建测试输出文件夹: {run_dir}")
    
    # 模型路径
    pretrain_model_path = "/home/zrz/AgentVNE/AgentVNE/pretrain_outputs/checkpoint_latest.pt"
    finetuning_model_path = "/home/zrz/AgentVNE/AgentVNE/finetuning_output_3/policy_network_latest.pth"
    
    # SN和Workflow路径
    sn_topology_path = os.path.join(script_dir, 'topo', 'SN_topology_2.json')
    workflow_path = os.path.join(script_dir, 'workflow_topo', 'workflow1_topo.json')
    
    # 参数设置
    device = 'cpu'
    num_wk = 2  # 处理的workflow数量
    
    print("="*80)
    print("概率矩阵测试脚本 - 多Workflow放置")
    print("="*80)
    
    # 1. 加载模型
    print("\n【步骤1】加载模型...")
    print(f"  预训练模型: {pretrain_model_path}")
    print(f"  微调模型: {finetuning_model_path}")
    
    pretrain_policy = load_model(pretrain_model_path, device=device)
    finetuning_policy = load_model(finetuning_model_path, device=device)
    
    # 2. 创建环境
    print("\n【步骤2】创建环境...")
    print(f"  SN拓扑: {sn_topology_path}")
    print(f"  Workflow: {workflow_path}")
    
    env = SimuVNEEnv(
        sn_topology_path=sn_topology_path,
        device=device,
        penalty=-150.0,
        max_arrived_tasks=num_wk
    )
    env.reset()
    
    # 获取SN容量用于VN特征归一化
    sn_capacity = env.get_sn_max_capacity()
    
    # 创建WorkflowGenerator
    workflow_types = {
        'workflow1': workflow_path
    }
    wf_gen = WorkflowGenerator(
        workflow_types=workflow_types,
        arrival_rate=1,
        mean_lifetime=2000.0,
        seed=42,
        sn_capacity_for_norm=sn_capacity
    )
    
    print(f"  ✓ 环境创建成功")
    print(f"  将处理 {num_wk} 个workflow")
    
    # 3. 处理多个workflow
    print("\n【步骤3】处理workflow...")
    
    accepted_count = 0
    rejected_count = 0
    
    for wk_idx in range(num_wk):
        print(f"\n--- 处理第 {wk_idx + 1}/{num_wk} 个workflow ---")
        
        # 加载workflow
        vn = wf_gen.load_workflow_graph('workflow1')
        print(f"  Workflow加载成功: {vn.x.size(0)} 个VN节点")
        
        # 获取当前SN状态
        sn_state = env.get_sn_state()
        
        # 【重要】如果是最后一个workflow，需要保存env的资源状态
        # 因为place_workflow会修改env中的资源，但计算概率矩阵需要使用放置前的状态
        if wk_idx == num_wk - 1:
            # 保存env中每个SN节点的资源状态（用于后续计算概率矩阵）
            saved_sn_resources = {}
            for sn_node_id in env.G_sn.nodes():
                node = env.G_sn.nodes[sn_node_id]
                saved_sn_resources[sn_node_id] = {
                    'cpu_res': node['cpu_res'],
                    'mem_res': node['mem_res'],
                    'disk_res': node['disk_res'],
                }
        
        # 使用微调模型进行放置（使用概率矩阵降序生成优先级列表）
        is_last_workflow = (wk_idx == num_wk - 1)
        verbose_placement = is_last_workflow  # 最后一个workflow时打印详细信息
        mapping, success, finetuning_probs_tensor = place_workflow(
            finetuning_policy, vn, sn_state, env, device=device, k_hop=1, verbose=verbose_placement
        )
        
        # 如果是最后一个workflow，计算并打印两个模型的概率矩阵
        if is_last_workflow:
            print(f"\n  【最后一个workflow】计算两个模型的概率矩阵...")
            
            # 【重要】恢复env的资源状态，确保计算概率矩阵时使用与place_workflow相同的状态
            for sn_node_id, resources in saved_sn_resources.items():
                node = env.G_sn.nodes[sn_node_id]
                node['cpu_res'] = resources['cpu_res']
                node['mem_res'] = resources['mem_res']
                node['disk_res'] = resources['disk_res']
            
            # 应用bias到SN特征（与放置时使用相同的状态）
            sn_with_bias = apply_bias_to_sn_features(sn_state, env)
            
            # 计算预训练模型的概率矩阵
            pretrain_probs = compute_probability_matrix(pretrain_policy, vn, sn_with_bias, device=device)
            
            # 微调模型的概率矩阵（从place_workflow返回）
            finetuning_probs = finetuning_probs_tensor.cpu().numpy()
            
            # 获取节点名称
            sn_node_list = sorted(env.G_sn.nodes())
            sn_node_names = [f"SN{sn_id}" for sn_id in sn_node_list]
            vn_node_names = [f"VN{i}" for i in range(vn.x.size(0))]
            
            # 打印概率矩阵
            print_probability_matrix(pretrain_probs, "预训练模型", vn_node_names, sn_node_names)
            print_probability_matrix(finetuning_probs, "微调模型", vn_node_names, sn_node_names)
            
            # 保存概率矩阵到带时间戳的文件夹
            pretrain_output_path = os.path.join(run_dir, 'pretrain_prob_matrix.txt')
            save_probability_matrix(pretrain_probs, "预训练模型", pretrain_output_path, vn_node_names, sn_node_names)
            
            finetuning_output_path = os.path.join(run_dir, 'finetuning_prob_matrix.txt')
            save_probability_matrix(finetuning_probs, "微调模型", finetuning_output_path, vn_node_names, sn_node_names)
            
            # 保存差值矩阵
            diff_probs = finetuning_probs - pretrain_probs
            diff_output_path = os.path.join(run_dir, 'prob_matrix_diff.txt')
            save_probability_matrix(diff_probs, "概率矩阵差值（微调-预训练）", diff_output_path, vn_node_names, sn_node_names)
            
            # 保存为numpy格式
            np_output_path = os.path.join(run_dir, 'prob_matrices.npz')
            np.savez(np_output_path, 
                     pretrain=pretrain_probs,
                     finetuning=finetuning_probs,
                     diff=diff_probs)
            print(f"  ✓ NumPy格式已保存到: {np_output_path}")
        
        # 检查放置结果
        if success and len(mapping) == vn.x.size(0):
            # 检查路径是否存在
            vn_paths = env._compute_paths_and_bw_demand(vn, mapping)
            if vn_paths is not None:
                # 添加到活跃workflow列表
                lifetime = wf_gen.sample_lifetime()
                expire_time = env.current_time + lifetime
                env.active_workflows.append({
                    'vn': vn,
                    'mapping': mapping,
                    'paths': vn_paths,
                    'expire_time': expire_time,
                    'task_id': wk_idx,
                })
                env.accepted_count += 1
                accepted_count += 1
                print(f"  ✓ Workflow {wk_idx + 1} 放置成功")
            else:
                # 路径不存在，需要回滚资源
                rollback_history = [(sn_id, vn_idx) for vn_idx, sn_id in mapping.items()]
                rollback_resource_deductions(env, rollback_history, vn, verbose=False)
                rejected_count += 1
                print(f"  ✗ Workflow {wk_idx + 1} 放置失败（路径不存在）")
        else:
            rejected_count += 1
            print(f"  ✗ Workflow {wk_idx + 1} 放置失败（资源不足）")
        
        # 推进时间（模拟时间流逝）
        env.step_time(time_delta=1.0)
    
    # 4. 打印总结
    print("\n" + "="*80)
    print("【总结】")
    print("="*80)
    print(f"  总workflow数: {num_wk}")
    print(f"  成功放置: {accepted_count}")
    print(f"  失败: {rejected_count}")
    print(f"  接受率: {accepted_count/num_wk:.2%}")
    print(f"  当前活跃workflow数: {len(env.active_workflows)}")
    print(f"\n所有结果已保存到: {run_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
