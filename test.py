#!/usr/bin/env python3
"""
测试脚本：加载fine_tuning模型，生成任务并放置
打印：概率矩阵、采样动作、放置前后SN剩余容量
"""
import os
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple
from torch.distributions import Categorical
from torch_geometric.data import Data

from model import SimuVNE
from fine_tuning import ValueNet, PPOAgent
from env import SimuVNEEnv, WorkflowGenerator


def print_probability_matrix(probs_matrix, vn_size, sn_size):
    """打印概率矩阵（格式化显示）"""
    print("  【概率矩阵】 (VN节点 × SN节点)")
    print("  " + "".join([f"SN{i:2d}  " for i in range(min(sn_size, 10))]))
    for i in range(vn_size):
        row_str = f"  VN{i}: "
        for j in range(min(sn_size, 10)):
            row_str += f"{probs_matrix[i, j].item():.3f} "
        if sn_size > 10:
            row_str += "..."
        print(row_str)
    if vn_size > 10:
        print("  ...")


def print_sn_residual(env, title="SN剩余容量"):
    """打印SN节点的剩余资源"""
    print(f"  【{title}】")
    for n in sorted(env.G_sn.nodes)[:10]:  # 只显示前10个节点
        nd = env.G_sn.nodes[n]
        print(f"    节点{n}: CPU={nd['cpu_res']:.2f}, Mem={nd['mem_res']:.2f}, Disk={nd['disk_res']:.2f}")
    if len(env.G_sn.nodes) > 10:
        print(f"    ... (共 {len(env.G_sn.nodes)} 个节点)")


def generate_priority_lists_deterministic(probs_matrix: torch.Tensor) -> List[List[int]]:
    """从概率矩阵生成优先级列表（测试时：按概率值降序排序，不采样）"""
    N_v, N_s = probs_matrix.shape
    priority_lists = []
    
    for i in range(N_v):
        probs = probs_matrix[i].cpu().numpy()  # [N_s]
        # 按概率值降序排序，得到SN节点索引
        sorted_indices = np.argsort(probs)[::-1]  # 降序
        priority_lists.append(sorted_indices.tolist())
    
    return priority_lists


def get_vn_neighbors(vn: Data) -> Dict[int, Set[int]]:
    """获取VN节点的邻居关系（无向，即使边是有向的）"""
    neighbors = {i: set() for i in range(vn.x.size(0))}
    edge_index = vn.edge_index
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        neighbors[u].add(v)
        neighbors[v].add(u)  # 即使边是有向的，也互为邻居
    return neighbors


def get_sn_k_hop_neighbors(env: SimuVNEEnv, sn_node_id: int, k: int) -> Set[int]:
    """获取SN节点的k跳邻居（包括k跳内的所有节点）"""
    if k == 0:
        return {sn_node_id}
    paths = nx.single_source_shortest_path_length(env.G_sn, sn_node_id, cutoff=k)
    return set(paths.keys())


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
        if verbose:
            print(f"        [资源检查] VN节点{vn_node_idx} → SN节点{sn_node_id}: CPU不足 (需求={cpu_demand:.3f}, 可用={sn_node['cpu_res']:.3f})")
        return False
    if mem_demand > sn_node['mem_res'] + 1e-9:
        if verbose:
            print(f"        [资源检查] VN节点{vn_node_idx} → SN节点{sn_node_id}: MEM不足 (需求={mem_demand:.3f}, 可用={sn_node['mem_res']:.3f})")
        return False
    if disk_demand > sn_node['disk_res'] + 1e-9:
        if verbose:
            print(f"        [资源检查] VN节点{vn_node_idx} → SN节点{sn_node_id}: DISK不足 (需求={disk_demand:.3f}, 可用={sn_node['disk_res']:.3f})")
        return False
    
    # 立即扣减资源
    sn_node['cpu_res'] -= cpu_demand
    sn_node['mem_res'] -= mem_demand
    sn_node['disk_res'] -= disk_demand
    
    if verbose:
        print(f"        [资源扣减] VN节点{vn_node_idx} → SN节点{sn_node_id}")
        print(f"          需求: CPU={cpu_demand:.3f}, MEM={mem_demand:.3f}, DISK={disk_demand:.3f}")
        print(f"          扣减后剩余: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
    
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
    if verbose:
        print(f"        回滚 {len(deduction_history)} 个节点的资源扣减:")
    
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
        
        if verbose:
            print(f"          恢复: VN节点{vn_node_idx} → SN节点{sn_node_id} (CPU={cpu_restore:.3f}, MEM={mem_restore:.3f}, DISK={disk_restore:.3f})")
            print(f"            SN节点{sn_node_id}剩余资源: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")


def place_with_bfs_strategy(vn: Data, sn_state: Data, env: SimuVNEEnv, 
                            probs_matrix: torch.Tensor, k_hop: int = 1, verbose: bool = False) -> Tuple[Dict[int, int], float]:
    """
    使用BFS扩展策略进行放置（测试版本：按概率值排序，立即扣减资源）
    
    Args:
        vn: VN图数据
        sn_state: SN状态数据
        env: 环境对象
        probs_matrix: 概率矩阵 [N_v, N_s]
        k_hop: k跳邻居参数（默认1）
        verbose: 是否打印详细信息
    
    Returns:
        (mapping, logprob): 节点映射和对数概率
    """
    N_v, N_s = probs_matrix.shape
    device = probs_matrix.device
    
    # 1. 生成优先级列表（按概率值降序）
    priority_lists = generate_priority_lists_deterministic(probs_matrix)
    
    if verbose:
        print(f"    【步骤1】生成优先级列表（按概率值降序）:")
        sn_node_list = sorted(env.G_sn.nodes())
        for i in range(N_v):
            priority_sn_ids = [sn_node_list[idx] for idx in priority_lists[i]]
            priority_probs = [float(probs_matrix[i][idx].item()) for idx in priority_lists[i]]
            if len(priority_sn_ids) > 10:
                print(f"      VN节点{i}: 优先级序列 = {priority_sn_ids[:10]}... (共{len(priority_sn_ids)}个)")
            else:
                print(f"      VN节点{i}: 优先级序列 = {priority_sn_ids}")
    
    # 2. 获取VN邻居关系
    vn_neighbors = get_vn_neighbors(vn)
    
    # 3. 计算VN节点度
    vn_degrees = {i: len(vn_neighbors[i]) for i in range(N_v)}
    
    # 4. 计算VN节点资源需求（用于排序）
    vn_resource_demands = {}
    for i in range(N_v):
        feats = vn.x[i]
        vn_resource_demands[i] = float(feats[0].item() + feats[1].item() + feats[2].item())
    
    if verbose:
        print(f"\n    【步骤2】VN节点资源需求:")
        for i in range(N_v):
            feats = vn.x[i]
            cpu_norm = float(feats[0].item())
            mem_norm = float(feats[1].item())
            disk_norm = float(feats[2].item())
            cpu_abs = cpu_norm * (env._sn_max_capacity['cpu_max'] + 1e-8)
            mem_abs = mem_norm * (env._sn_max_capacity['mem_max'] + 1e-8)
            disk_abs = disk_norm * (env._sn_max_capacity['disk_max'] + 1e-8)
            print(f"      VN节点{i}: 归一化=(CPU:{cpu_norm:.4f}, MEM:{mem_norm:.4f}, DISK:{disk_norm:.4f}), "
                  f"绝对=(CPU:{cpu_abs:.3f}, MEM:{mem_abs:.3f}, DISK:{disk_abs:.3f}), 度={vn_degrees[i]}")
    
    # 5. 获取SN节点ID列表（用于索引映射）
    sn_node_list = sorted(env.G_sn.nodes())
    
    if verbose:
        print(f"\n    【步骤3】SN节点当前资源状态:")
        for sn_id in sn_node_list[:10]:
            sn_node = env.G_sn.nodes[sn_id]
            print(f"      SN节点{sn_id}: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
    
    # 6. 选择第一个VN节点（资源占用最大）
    first_vn = max(range(N_v), key=lambda i: vn_resource_demands[i])
    
    if verbose:
        print(f"\n    【步骤4】选择第一个VN节点:")
        print(f"      选择: VN节点{first_vn} (资源需求最大: {vn_resource_demands[first_vn]:.4f})")
    
    # 7. 放置第一个VN节点（按优先级顺序逐一尝试，成功则立即扣减资源）
    mapping: Dict[int, int] = {}
    resource_deduction_history: List[Tuple[int, int]] = []
    placed_first = False
    tried_count_first = 0
    
    for first_sn_idx in priority_lists[first_vn]:
        first_sn_id = sn_node_list[first_sn_idx]
        tried_count_first += 1
        if verbose:
            print(f"      尝试优先级第{tried_count_first}个: SN节点{first_sn_id} (索引{first_sn_idx})")
        if check_and_deduct_resource(env, first_sn_id, first_vn, vn, verbose=verbose):
            mapping[first_vn] = first_sn_id
            resource_deduction_history.append((first_sn_id, first_vn))
            placed_first = True
            if verbose:
                print(f"      ✓ 成功放置: VN节点{first_vn} → SN节点{first_sn_id}")
            break
        else:
            if verbose:
                print(f"      ✗ 失败: SN节点{first_sn_id}资源不足")
    
    if not placed_first:
        if verbose:
            print(f"      ✗ 无法在任一SN节点上放置第一个VN节点，任务失败")
        return {}, 0.0
    
    # 8. BFS扩展放置（实时扣减资源）
    placed_vn: Set[int] = {first_vn}
    queue = [first_vn]
    bfs_round = 0
    
    if verbose:
        print(f"\n    【步骤5】BFS扩展放置:")
    
    while queue and len(placed_vn) < N_v:
        bfs_round += 1
        new_placed: List[int] = []
        
        if verbose:
            print(f"\n      --- BFS轮次 {bfs_round} ---")
            print(f"      队列: {queue} (已放置: {sorted(placed_vn)})")
        
        for vi in queue:
            vi_sn_id = mapping[vi]
            
            if verbose:
                print(f"\n      处理队列节点: VN节点{vi} (当前在SN节点{vi_sn_id})")
            
            # 找到vi的未放置邻居
            unplaced_neighbors = [u for u in vn_neighbors[vi] if u not in placed_vn]
            
            if verbose:
                print(f"        未放置的邻居VN节点: {unplaced_neighbors}")
            
            # 对每个邻居尝试放置
            for u in unplaced_neighbors:
                if verbose:
                    print(f"\n        尝试放置邻居: VN节点{u}")
                
                # 策略1: 尝试放在同一个SN节点上
                if verbose:
                    print(f"        策略1: 尝试放在同一SN节点{vi_sn_id}上")
                
                if check_and_deduct_resource(env, vi_sn_id, u, vn, verbose=verbose):
                    mapping[u] = vi_sn_id
                    placed_vn.add(u)
                    new_placed.append(u)
                    resource_deduction_history.append((vi_sn_id, u))
                    if verbose:
                        print(f"        ✓ 成功放置: VN节点{u} → SN节点{vi_sn_id}")
                    continue
                else:
                    if verbose:
                        print(f"        ✗ 失败: SN节点{vi_sn_id}资源不足")
                
                # 策略2: 在k跳邻居中找
                k = 1
                max_k = len(sn_node_list)
                placed = False
                
                if verbose:
                    print(f"        策略2: 在k跳邻居中搜索 (从k=1开始, 最大k={max_k})")
                
                while k <= max_k and not placed:
                    k_hop_neighbors = get_sn_k_hop_neighbors(env, vi_sn_id, k)
                    
                    if verbose:
                        print(f"          k={k}: k跳邻居SN节点 = {sorted(k_hop_neighbors)}")
                    
                    tried_count = 0
                    # 按优先级列表顺序尝试
                    for sn_idx in priority_lists[u]:
                        sn_id = sn_node_list[sn_idx]
                        if sn_id in k_hop_neighbors:
                            tried_count += 1
                            if verbose:
                                print(f"            尝试优先级第{tried_count}个: SN节点{sn_id} (索引{sn_idx})", end=' ')
                            
                            if check_and_deduct_resource(env, sn_id, u, vn, verbose=verbose):
                                mapping[u] = sn_id
                                placed_vn.add(u)
                                new_placed.append(u)
                                resource_deduction_history.append((sn_id, u))
                                placed = True
                                if verbose:
                                    print(f"→ ✓ 成功放置!")
                                break
                            else:
                                if verbose:
                                    print(f"→ ✗ 资源不足")
                    
                    if placed:
                        break
                    
                    if verbose:
                        print(f"          k={k}跳内无法放置，扩展到k={k+1}")
                    
                    k += 1
                
                if not placed:
                    # 无法放置，需要回滚所有资源扣减
                    if verbose:
                        print(f"\n        ⚠️ 放置失败: VN节点{u} 在所有k跳内都无法找到合适位置")
                        print(f"        开始回滚资源扣减...")
                    rollback_resource_deductions(env, resource_deduction_history, vn, verbose=verbose)
                    return {}, 0.0
        
        # 更新队列：按度降序；若度相同，则按资源需求降序
        queue = sorted(
            new_placed,
            key=lambda i: (vn_degrees[i], vn_resource_demands[i]),
            reverse=True
        )
        
        if verbose:
            if queue:
                print(f"\n      本轮新放置: {sorted(new_placed)}")
                print(f"      下一轮队列: {queue}")
            else:
                print(f"\n      本轮无新放置，BFS结束")
    
    # 检查是否所有节点都已放置
    if len(placed_vn) < N_v:
        if verbose:
            print(f"\n        ⚠️ 部分节点未放置 ({len(placed_vn)}/{N_v})，回滚资源扣减...")
        rollback_resource_deductions(env, resource_deduction_history, vn, verbose=verbose)
        return {}, 0.0
    
    if verbose:
        print(f"\n    【步骤6】放置完成:")
        print(f"      已放置VN节点: {sorted(placed_vn)} / {N_v}")
        print(f"      最终映射:")
        for vn_idx in sorted(mapping.keys()):
            print(f"        VN节点{vn_idx} → SN节点{mapping[vn_idx]}")
    
    # 9. 计算logprob
    logprob_sum = 0.0
    sn_id_to_idx = {sn_id: idx for idx, sn_id in enumerate(sn_node_list)}
    for vn_idx, sn_id in mapping.items():
        probs = probs_matrix[vn_idx]
        cat = Categorical(probs=probs)
        sn_idx = sn_id_to_idx[sn_id]
        logprob_sum += float(cat.log_prob(torch.tensor(sn_idx, device=device)).item())
    
    return mapping, logprob_sum


def test_placement_with_model(model_path=None, 
                              arrival_rate: float = 1.0,
                              mean_lifetime: float = 10.0,
                              max_arrived_tasks: int = 10,
                              max_time_steps: int = 1000,
                              seed: int = 42):
    """
    测试任务放置流程（时间驱动版本）
    
    Args:
        model_path: 模型checkpoint路径，None则使用随机初始化
        arrival_rate: 泊松到达率（每个时间单位的任务到达率）
        mean_lifetime: 平均生存时间（指数分布）
        max_arrived_tasks: 最大到达任务数
        max_time_steps: 最大时间步数
        seed: 随机种子
    """
    print("=" * 80)
    print("测试：加载模型并进行任务放置")
    print("=" * 80)
    
    # 配置
    sn_path = '/home/zrz/SimuVNE/topo/SN_topology.json'
    workflow_types = {
        'workflow1': '/home/zrz/SimuVNE/workflow_topo/workflow1_topo.json',
    }
    device = 'cpu'
    
    # 创建环境
    print(f"\n【1. 初始化环境】")
    env = SimuVNEEnv(
        sn_topology_path=sn_path,
        device=device,
        penalty=-150.0,
        max_arrived_tasks=max_arrived_tasks
    )
    env.reset()
    env.current_time = 0.0  # 确保时间从0开始
    sn_capacity = env.get_sn_max_capacity()
    
    print(f"  SN拓扑: {len(env.G_sn.nodes)} 个节点, {len(env.G_sn.edges)} 条边")
    print(f"  SN最大容量: CPU={sn_capacity['cpu_max']}, Mem={sn_capacity['mem_max']}, Disk={sn_capacity['disk_max']}")
    print(f"  任务参数: 到达率={arrival_rate}, 平均生存时间={mean_lifetime}, 最大任务数={max_arrived_tasks}, 最大时间步={max_time_steps}")
    
    # 创建工作流生成器
    wf_gen = WorkflowGenerator(
        workflow_types=workflow_types,
        arrival_rate=arrival_rate,
        mean_lifetime=mean_lifetime,
        seed=seed,
        sn_capacity_for_norm=sn_capacity
    )
    
    # 加载或创建模型
    print(f"\n【2. 加载模型】")
    policy = SimuVNE()
    value_net = ValueNet()
    
    if model_path:
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            # 处理不同的checkpoint格式
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                if 'model_config' in ckpt:
                    print(f"  模型配置: {ckpt['model_config']}")
            else:
                # 如果checkpoint本身就是state_dict
                state_dict = ckpt
            
            policy.load_state_dict(state_dict, strict=False)
            print(f"  ✓ 已成功加载策略网络: {model_path}")
            
            # 尝试加载价值网络（如果存在）
            value_model_path = model_path.replace('policy_network.pth', 'value_network.pth')
            if os.path.exists(value_model_path):
                try:
                    value_ckpt = torch.load(value_model_path, map_location='cpu')
                    if isinstance(value_ckpt, dict) and 'model_state_dict' in value_ckpt:
                        value_state_dict = value_ckpt['model_state_dict']
                    else:
                        value_state_dict = value_ckpt
                    value_net.load_state_dict(value_state_dict, strict=False)
                    print(f"  ✓ 已成功加载价值网络: {value_model_path}")
                except Exception as e:
                    print(f"  ⚠ 加载价值网络失败: {e}，使用随机初始化的价值网络")
            else:
                print(f"  → 未找到价值网络文件，使用随机初始化的价值网络")
        except Exception as e:
            print(f"  ✗ 加载模型失败: {e}")
            print(f"  → 使用随机初始化的模型")
    else:
        print(f"  → 使用随机初始化的模型")
    
    agent = PPOAgent(policy, value_net, device=device)
    policy.eval()  # 设置为评估模式
    
    # 打印初始SN状态
    print(f"\n【3. 初始SN状态】")
    print_sn_residual(env, "SN初始剩余容量")
    
    # 开始放置任务（时间驱动版本）
    print(f"\n{'=' * 80}")
    print(f"【4. 开始任务放置】 (时间驱动: 到达率={arrival_rate}, 最大任务数={max_arrived_tasks}, 最大时间步={max_time_steps})")
    print(f"{'=' * 80}")
    
    placed_count = 0
    failed_count = 0
    time_step = 0
    
    while time_step < max_time_steps and not env.is_done():
        # 1) 推进时间，移除到期任务
        env.step_time(time_delta=1.0)
        
        # 2) 检查是否有任务到达
        has_arrival = wf_gen.check_arrival(time_unit=1.0)
        
        if has_arrival and not env.is_done():
            # 任务到达
        wf_type = wf_gen.sample_workflow_type()
        vn = wf_gen.load_workflow_graph(wf_type)
        lifetime = wf_gen.sample_lifetime()
            task_id = env.arrived_count
            env.arrived_count += 1
            
            print(f"\n{'-' * 80}")
            print(f"[t={env.current_time:.1f}] 任务 #{task_id} 到达")
            print(f"{'-' * 80}")
        
        print(f"  任务类型: {wf_type}")
        print(f"  VN节点数: {vn.x.size(0)}")
        print(f"  生存时间: {lifetime:.1f}")
        
        # 打印VN需求（归一化）
        print(f"\n  【VN节点需求】 (归一化)")
        for i in range(min(vn.x.size(0), 6)):
            feats = vn.x[i]
            print(f"    节点{i}: cpu={feats[0].item():.3f}, mem={feats[1].item():.3f}, disk={feats[2].item():.3f}")
        
        # 获取当前SN状态
        sn_state = env.get_sn_state()
        
        # 策略网络推理
        print(f"\n  【策略网络推理】")
        with torch.no_grad():
            vn_input = vn.to(device)
            sn_input = sn_state.to(device)
            
            # 获取概率矩阵
            probs_matrix = policy(vn_input, sn_input)
            print(f"  输出形状: {probs_matrix.shape} (VN节点 × SN节点)")
        
        # 打印概率矩阵
        print_probability_matrix(probs_matrix, vn.x.size(0), sn_state.x.size(0))
        
        # 打印放置前SN状态
        print(f"\n  【放置前SN状态】")
        print_sn_residual(env, "放置前SN剩余容量")
        
            # 使用新的BFS放置策略（资源已在策略中扣减）
            print(f"\n  【BFS放置策略】")
            print(f"  {'='*70}")
            mapping, logprob = place_with_bfs_strategy(vn, sn_state, env, probs_matrix, k_hop=1, verbose=True)
            print(f"  {'='*70}")
            print(f"  最终映射: {mapping}")
            print(f"  Log概率: {logprob:.4f}")
            
            # 计算状态价值
            with torch.no_grad():
                value = agent.value_net(vn.to(device), sn_state.to(device))
            print(f"  状态价值: {value.item():.4f}")
            
            # 检查是否成功放置（所有节点都已映射，资源已在place_with_bfs_strategy中扣减）
            print(f"\n  【尝试放置】")
            if len(mapping) == vn.x.size(0):
                # 所有节点都已放置，资源已在place_with_bfs_strategy中扣减，只需要检查路径并添加工作流
                vn_paths = env._compute_paths_and_bw_demand(vn, mapping)
                if vn_paths is None:
                    # 路径不存在，需要回滚资源
                    rollback_history = [(sn_id, vn_idx) for vn_idx, sn_id in mapping.items()]
                    rollback_resource_deductions(env, rollback_history, vn, verbose=False)
                    success, r_t = False, env.penalty
                    print(f"  ✗ 放置失败: 路径不存在")
                    print(f"    penalty = {r_t:.3f}")
                    failed_count += 1
                else:
                    # 路径存在，添加工作流（资源已扣减，不需要再次扣减）
                    expire_time = env.current_time + lifetime
                    env.active_workflows.append({
                        'vn': vn,
                        'mapping': mapping,
                        'paths': vn_paths,
                        'expire_time': expire_time,
                        'task_id': task_id,
                    })
                    env.accepted_count += 1
                    r_t = env._compute_rt()
                    success = True
                    print(f"  ✓ 放置成功")
                print(f"    r_t = {r_t:.3f}")
                placed_count += 1
            else:
                # 部分节点未放置，资源已在place_with_bfs_strategy中回滚
                success, r_t = False, env.penalty
                print(f"  ✗ 放置失败: 部分节点未放置 ({len(mapping)}/{vn.x.size(0)})")
            print(f"    penalty = {r_t:.3f}")
            failed_count += 1
        
        # 打印放置后SN状态
        print(f"\n  【放置后SN状态】")
        print_sn_residual(env, "放置后SN剩余容量")
        print(f"  当前存活任务数: {len(env.active_workflows)}")
            print(f"  [t={env.current_time:.1f}] 任务 #{task_id} {'✓成功' if success else '✗失败'} (r_t={r_t:.3f}, 存活任务数:{len(env.active_workflows)})")
        
        time_step += 1
    
    # 统计结果
    print(f"\n{'=' * 80}")
    print(f"【5. 测试结果统计】")
    print(f"{'=' * 80}")
    print(f"  时间步数: {time_step} / {max_time_steps}")
    print(f"  到达任务数: {env.arrived_count} / {max_arrived_tasks}")
    if env.arrived_count > 0:
        print(f"  成功放置: {placed_count} ({placed_count/env.arrived_count*100:.1f}%)")
        print(f"  放置失败: {failed_count} ({failed_count/env.arrived_count*100:.1f}%)")
    else:
        print(f"  成功放置: {placed_count}")
        print(f"  放置失败: {failed_count}")
    print(f"  最终存活任务数: {len(env.active_workflows)}")
    print(f"  最终时间: {env.current_time:.1f}")
    
    # 打印最终SN状态
    print(f"\n【6. 最终SN状态】")
    print_sn_residual(env, "最终SN剩余容量")
    
    print(f"\n{'=' * 80}")
    print("测试完成！")
    print(f"{'=' * 80}")


def find_latest_finetuning_model(finetuning_dir='/home/zrz/SimuVNE/finetuning_putput'):
    """
    查找最新的fine-tuning模型
    
    Args:
        finetuning_dir: fine-tuning输出目录
    
    Returns:
        最新的模型路径，如果不存在则返回None
    """
    import os
    import glob
    
    if not os.path.exists(finetuning_dir):
        return None
    
    runs = sorted(glob.glob(f"{finetuning_dir}/run_*/policy_network.pth"))
    if runs:
        return runs[-1]  # 返回最新的模型
    return None


if __name__ == '__main__':
    # 优先使用fine-tuning模型
    finetuning_dir = '/home/zrz/SimuVNE/finetuning_putput'
    latest_finetuning_model = find_latest_finetuning_model(finetuning_dir)
    
    if latest_finetuning_model:
        print("\n" + "=" * 80)
        print("【使用Fine-tuning模型进行测试】")
        print("=" * 80)
        print(f"模型路径: {latest_finetuning_model}")
        test_placement_with_model(
            model_path=latest_finetuning_model,
            arrival_rate=1.0,
            mean_lifetime=10.0,
            max_arrived_tasks=10,
            max_time_steps=1000,
            seed=42
        )
    else:
        # 如果找不到fine-tuning模型，尝试使用预训练模型
        pretrained_model = '/home/zrz/SimuVNE/pretrain_outputs/checkpoint_best.pt'
        if os.path.exists(pretrained_model):
            print("\n" + "=" * 80)
            print("【未找到Fine-tuning模型，使用预训练模型】")
            print("=" * 80)
            print(f"模型路径: {pretrained_model}")
            test_placement_with_model(
                model_path=pretrained_model,
                arrival_rate=1.0,
                mean_lifetime=10.0,
                max_arrived_tasks=10,
                max_time_steps=1000,
                seed=42
            )
        else:
            # 如果预训练模型也不存在，使用随机初始化模型
            print("\n" + "=" * 80)
            print("【未找到Fine-tuning和预训练模型，使用随机初始化模型】")
            print("=" * 80)
            print("警告: 使用随机初始化的模型，测试结果可能不准确")
            test_placement_with_model(
                model_path=None,
                arrival_rate=1.0,
                mean_lifetime=10.0,
                max_arrived_tasks=10,
                max_time_steps=1000,
                seed=42
            )

