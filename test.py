#!/usr/bin/env python3
"""
测试脚本：加载fine_tuning模型，生成任务并放置
打印：概率矩阵、采样动作、放置前后SN剩余容量
"""
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


def check_sn_resource(env: SimuVNEEnv, sn_node_id: int, vn_node_idx: int, vn: Data,
                     temp_mapping: Optional[Dict[int, int]] = None) -> bool:
    """检查SN节点是否有足够资源放置VN节点"""
    sn_node = env.G_sn.nodes[sn_node_id]
    vn_feats = vn.x[vn_node_idx]
    
    # 计算绝对资源需求
    cpu_demand = float(vn_feats[0].item()) * (env._sn_max_capacity['cpu_max'] + 1e-8)
    mem_demand = float(vn_feats[1].item()) * (env._sn_max_capacity['mem_max'] + 1e-8)
    disk_demand = float(vn_feats[2].item()) * (env._sn_max_capacity['disk_max'] + 1e-8)
    
    # 计算当前SN节点的可用资源（考虑临时映射中已放置的节点）
    available_cpu = sn_node['cpu_res']
    available_mem = sn_node['mem_res']
    available_disk = sn_node['disk_res']
    
    if temp_mapping:
        # 虚拟扣减当前轮已放置在该SN节点上的VN节点的资源
        for vn_idx, sn_id in temp_mapping.items():
            if sn_id == sn_node_id:
                vn_feats_temp = vn.x[vn_idx]
                available_cpu -= float(vn_feats_temp[0].item()) * (env._sn_max_capacity['cpu_max'] + 1e-8)
                available_mem -= float(vn_feats_temp[1].item()) * (env._sn_max_capacity['mem_max'] + 1e-8)
                available_disk -= float(vn_feats_temp[2].item()) * (env._sn_max_capacity['disk_max'] + 1e-8)
    
    # 检查剩余资源
    if cpu_demand > available_cpu + 1e-9:
        return False
    if mem_demand > available_mem + 1e-9:
        return False
    if disk_demand > available_disk + 1e-9:
        return False
    return True


def place_with_bfs_strategy(vn: Data, sn_state: Data, env: SimuVNEEnv, 
                            probs_matrix: torch.Tensor, k_hop: int = 1) -> Tuple[Dict[int, int], float]:
    """
    使用BFS扩展策略进行放置（测试版本：按概率值排序）
    
    Args:
        vn: VN图数据
        sn_state: SN状态数据
        env: 环境对象
        probs_matrix: 概率矩阵 [N_v, N_s]
        k_hop: k跳邻居参数（默认1）
    
    Returns:
        (mapping, logprob): 节点映射和对数概率
    """
    N_v, N_s = probs_matrix.shape
    device = probs_matrix.device
    
    # 1. 生成优先级列表（按概率值降序）
    priority_lists = generate_priority_lists_deterministic(probs_matrix)
    
    # 2. 获取VN邻居关系
    vn_neighbors = get_vn_neighbors(vn)
    
    # 3. 计算VN节点度
    vn_degrees = {i: len(vn_neighbors[i]) for i in range(N_v)}
    
    # 4. 计算VN节点资源需求（用于排序）
    vn_resource_demands = {}
    for i in range(N_v):
        feats = vn.x[i]
        vn_resource_demands[i] = float(feats[0].item() + feats[1].item() + feats[2].item())
    
    # 5. 获取SN节点ID列表（用于索引映射）
    sn_node_list = sorted(env.G_sn.nodes())
    
    # 6. 选择第一个VN节点（资源占用最大）
    first_vn = max(range(N_v), key=lambda i: vn_resource_demands[i])
    
    # 7. 放置第一个VN节点（按概率优先级依次尝试，使用资源可行性检查，不做实际扣减）
    mapping: Dict[int, int] = {}
    placed_first = False
    for first_sn_idx in priority_lists[first_vn]:
        first_sn_id = sn_node_list[first_sn_idx]
        # 对首个节点，当前轮临时映射为空，无需临时扣减
        if check_sn_resource(env, first_sn_id, first_vn, vn, temp_mapping=None):
            mapping[first_vn] = first_sn_id
            placed_first = True
            break
    if not placed_first:
        # 无法在任何SN上放置首个节点，直接返回（失败的mapping，后续env.try_place_task将失败）
        return {}, 0.0
    
    # 8. BFS扩展放置
    placed_vn: Set[int] = {first_vn}
    queue = [first_vn]
    
    while queue and len(placed_vn) < N_v:
        new_placed: List[int] = []
        
        for vi in queue:
            vi_sn_id = mapping[vi]
            
            # 找到vi的未放置邻居
            unplaced_neighbors = [u for u in vn_neighbors[vi] if u not in placed_vn]
            
            # 对每个邻居尝试放置
            for u in unplaced_neighbors:
                # 首先尝试放在同一个SN节点上
                # 注意：temp_mapping 应该只包含当前轮新放置但尚未扣减资源的节点
                # 由于资源在 env.try_place_task() 时才扣减，这里传入当前轮新放置的节点
                current_round_temp = {vn: sn for vn, sn in mapping.items() if vn in new_placed}
                if check_sn_resource(env, vi_sn_id, u, vn, temp_mapping=current_round_temp):
                    mapping[u] = vi_sn_id
                    placed_vn.add(u)
                    new_placed.append(u)
                    continue
                
                # 否则在k跳邻居中找
                k = 1
                # 允许扩展到整个SN网络（最大跳数为SN节点数）
                max_k = len(sn_node_list)
                placed = False
                
                while k <= max_k and not placed:
                    k_hop_neighbors = get_sn_k_hop_neighbors(env, vi_sn_id, k)
                    # 按优先级列表顺序尝试
                    # 同样，temp_mapping 只包含当前轮新放置的节点
                    current_round_temp = {vn: sn for vn, sn in mapping.items() if vn in new_placed}
                    for sn_idx in priority_lists[u]:
                        sn_id = sn_node_list[sn_idx]
                        if sn_id in k_hop_neighbors and check_sn_resource(env, sn_id, u, vn, temp_mapping=current_round_temp):
                            mapping[u] = sn_id
                            placed_vn.add(u)
                            new_placed.append(u)
                            placed = True
                            break
                    # 如果当前k跳内没有找到可放置节点，扩展到k+1跳
                    k += 1
        
        # 更新队列：按度降序；若度相同，则按资源需求降序
        queue = sorted(
            new_placed,
            key=lambda i: (vn_degrees[i], vn_resource_demands[i]),
            reverse=True
        )
    
    # 9. 计算logprob
    logprob_sum = 0.0
    sn_id_to_idx = {sn_id: idx for idx, sn_id in enumerate(sn_node_list)}
    for vn_idx, sn_id in mapping.items():
        probs = probs_matrix[vn_idx]
        cat = Categorical(probs=probs)
        sn_idx = sn_id_to_idx[sn_id]
        logprob_sum += float(cat.log_prob(torch.tensor(sn_idx, device=device)).item())
    
    return mapping, logprob_sum


def test_placement_with_model(model_path=None, num_tasks=5):
    """
    测试任务放置流程
    
    Args:
        model_path: 模型checkpoint路径，None则使用随机初始化
        num_tasks: 测试任务数量
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
        max_arrived_tasks=num_tasks
    )
    env.reset()
    sn_capacity = env.get_sn_max_capacity()
    
    print(f"  SN拓扑: {len(env.G_sn.nodes)} 个节点, {len(env.G_sn.edges)} 条边")
    print(f"  SN最大容量: CPU={sn_capacity['cpu_max']}, Mem={sn_capacity['mem_max']}, Disk={sn_capacity['disk_max']}")
    
    # 创建工作流生成器
    wf_gen = WorkflowGenerator(
        workflow_types=workflow_types,
        arrival_rate=1.0,  # 每个时间单位必定到达
        mean_lifetime=10.0,
        seed=42,
        sn_capacity_for_norm=sn_capacity
    )
    
    # 加载或创建模型
    print(f"\n【2. 加载模型】")
    policy = SimuVNE()
    value_net = ValueNet()
    
    if model_path:
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            policy.load_state_dict(state_dict, strict=False)
            print(f"  ✓ 已加载模型: {model_path}")
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
    
    # 开始放置任务
    print(f"\n{'=' * 80}")
    print(f"【4. 开始任务放置】 (共 {num_tasks} 个任务)")
    print(f"{'=' * 80}")
    
    placed_count = 0
    failed_count = 0
    
    for task_id in range(num_tasks):
        print(f"\n{'-' * 80}")
        print(f"任务 #{task_id}")
        print(f"{'-' * 80}")
        
        # 生成VN任务
        wf_type = wf_gen.sample_workflow_type()
        vn = wf_gen.load_workflow_graph(wf_type)
        lifetime = wf_gen.sample_lifetime()
        
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
        
        # 使用新的BFS放置策略
        print(f"\n  【BFS放置策略】")
        mapping, logprob = place_with_bfs_strategy(vn, sn_state, env, probs_matrix, k_hop=1)
        print(f"  映射: {mapping}")
        print(f"  Log概率: {logprob:.4f}")
        
        # 计算状态价值
        with torch.no_grad():
            value = agent.value_net(vn.to(device), sn_state.to(device))
        print(f"  状态价值: {value.item():.4f}")
        
        # 打印放置前SN状态
        print(f"\n  【放置前SN状态】")
        print_sn_residual(env, "放置前SN剩余容量")
        
        # 尝试放置
        print(f"\n  【尝试放置】")
        success, r_t = env.try_place_task(vn, mapping, lifetime, task_id)
        
        if success:
            print(f"  ✓ 放置成功")
            print(f"    r_t = {r_t:.3f}")
            placed_count += 1
        else:
            print(f"  ✗ 放置失败")
            print(f"    penalty = {r_t:.3f}")
            failed_count += 1
        
        # 打印放置后SN状态
        print(f"\n  【放置后SN状态】")
        print_sn_residual(env, "放置后SN剩余容量")
        print(f"  当前存活任务数: {len(env.active_workflows)}")
        
        # 推进时间（模拟任务过期）
        env.arrived_count += 1
        env.step_time(time_delta=2.0)  # 推进2个时间单位
    
    # 统计结果
    print(f"\n{'=' * 80}")
    print(f"【5. 测试结果统计】")
    print(f"{'=' * 80}")
    print(f"  总任务数: {num_tasks}")
    print(f"  成功放置: {placed_count} ({placed_count/num_tasks*100:.1f}%)")
    print(f"  放置失败: {failed_count} ({failed_count/num_tasks*100:.1f}%)")
    print(f"  最终存活任务数: {len(env.active_workflows)}")
    
    # 打印最终SN状态
    print(f"\n【6. 最终SN状态】")
    print_sn_residual(env, "最终SN剩余容量")
    
    print(f"\n{'=' * 80}")
    print("测试完成！")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    # 测试场景1: 使用随机初始化模型
    print("\n【场景1: 随机初始化模型】")
    test_placement_with_model(model_path=None, num_tasks=3)
    
    # 测试场景2: 使用预训练模型（如果存在）
    # pretrained_model = '/home/zrz/SimuVNE/pretrain_outputs/checkpoint_best.pt'
    # print("\n\n【场景2: 预训练模型】")
    # test_placement_with_model(model_path=pretrained_model, num_tasks=3)
    
    # 测试场景3: 使用fine-tuning模型（如果存在）
    # 查找最新的fine-tuning输出
    import os
    import glob
    finetuning_dir = '/home/zrz/SimuVNE/finetuning_putput'
    runs = sorted(glob.glob(f"{finetuning_dir}/run_*/policy_network.pth"))
    if runs:
        latest_model = runs[-1]
        print(f"\n\n【场景3: Fine-tuning模型】")
        print(f"最新模型: {latest_model}")
        test_placement_with_model(model_path=latest_model, num_tasks=3)
    else:
        print(f"\n未找到fine-tuning模型，跳过场景3")

