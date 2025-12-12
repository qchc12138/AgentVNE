#!/usr/bin/env python3
"""
GAL (Greedy Allocation Algorithm) - 对比基准算法

贪心放置策略：
1. 按VN节点的归一化需求强度(cpu+mem+disk)从大到小排序
2. 按SN节点的剩余资源(cpu+mem+disk)从大到小排序
3. 按顺序贪心放置：为每个VN节点选择能容纳它的最大剩余资源的SN节点

与pretrain/fine-tuning的关键区别：
- 任务生成、环境管理完全相同
- 放置策略不同：GAL使用贪心策略，而非神经网络决策
"""

import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import os

import torch
import numpy as np
from torch_geometric.data import Data

from env import SimuVNEEnv, WorkflowGenerator
from baselines.noderank_utils import compute_sn_noderank_from_graph


class GreedyAllocator:
    """贪心放置算法（基于 noderank，每次workflow到达时重新计算）"""
    
    def __init__(self, env: SimuVNEEnv):
        self.env = env
        # 创建 SN 节点 ID 到索引的映射（必须在noderank计算前创建，确保顺序一致）
        sn_node_list = sorted(env.G_sn.nodes())
        self.sn_node_to_idx = {node_id: idx for idx, node_id in enumerate(sn_node_list)}
    
    def _compute_noderank_from_residual_resources(self) -> np.ndarray:
        """
        基于剩余资源计算SN NodeRank。
        
        Returns:
            noderank: numpy 数组，基于剩余资源计算的NodeRank值
        """
        return compute_sn_noderank_from_graph(self.env.G_sn, use_residual_resources=True)
    
    def _get_vn_node_demand(self, vn: Data, vn_idx: int) -> Dict[str, float]:
        """获取VN节点的资源需求"""
        feats = vn.x[vn_idx]
        norm_cpu = float(feats[0].item())
        norm_mem = float(feats[1].item())
        norm_disk = float(feats[2].item())
        
        abs_cpu = norm_cpu * (self.env._sn_max_capacity['cpu_max'] + 1e-8)
        abs_mem = norm_mem * (self.env._sn_max_capacity['mem_max'] + 1e-8)
        abs_disk = norm_disk * (self.env._sn_max_capacity['disk_max'] + 1e-8)
        norm_demand = norm_cpu + norm_mem + norm_disk
        
        return {
            'abs_cpu': abs_cpu,
            'abs_mem': abs_mem,
            'abs_disk': abs_disk,
            'norm_demand': norm_demand,
        }
    
    def _bfs_vn_nodes(self, vn: Data, start_node: int, non_constraint_indices: List[int]) -> List[int]:
        """
        从起始节点开始，按广度优先遍历VN图，返回非约束节点的遍历顺序。
        
        Args:
            vn: VN图数据
            start_node: 起始节点（资源消耗最高的节点）
            non_constraint_indices: 非约束节点索引列表
        
        Returns:
            按BFS顺序排列的非约束节点索引列表
        """
        # 构建VN图的邻接列表
        vn_adjacency: Dict[int, List[int]] = {i: [] for i in range(vn.x.size(0))}
        if vn.edge_index.numel() > 0:
            for i in range(vn.edge_index.size(1)):
                u = int(vn.edge_index[0, i].item())
                v = int(vn.edge_index[1, i].item())
                if v not in vn_adjacency[u]:
                    vn_adjacency[u].append(v)
                if u not in vn_adjacency[v]:
                    vn_adjacency[v].append(u)
        
        # BFS遍历
        visited = set()
        queue = [start_node]
        visited.add(start_node)
        bfs_order = []
        
        while queue:
            current = queue.pop(0)
            if current in non_constraint_indices:
                bfs_order.append(current)
            
            # 添加未访问的邻居
            for neighbor in vn_adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # 如果还有未访问的非约束节点（可能是孤立节点），添加到末尾
        for vn_idx in non_constraint_indices:
            if vn_idx not in visited:
                bfs_order.append(vn_idx)
        
        return bfs_order
    
    def greedy_place(self, vn: Data) -> tuple[bool, Dict[int, int], float]:
        """
        贪心放置策略（基于剩余资源计算noderank，广度优先遍历VN节点）
        
        算法流程：
        1. 每次workflow到达时，基于剩余资源重新计算SN NodeRank
        2. 找到资源消耗最高的VN节点作为起始节点
        3. 按广度优先遍历VN图，依次放置VN节点
        4. 每个VN节点优先放在noderank最高的可用SN节点上
        
        Args:
            vn: VN图数据（特征已归一化）
        
        Returns:
            (success, mapping, r_t): 成功标志、节点映射、奖励
        """
        # 导入约束节点处理工具
        from tests.constraint_handler import separate_constraint_nodes, place_constraint_nodes
        
        # 1. 分离约束节点和非约束节点
        non_constraint_indices, constraint_indices, constraint_mapping = separate_constraint_nodes(vn)
        
        # 2. 基于剩余资源重新计算SN NodeRank（每次workflow到达时）
        sn_noderank = self._compute_noderank_from_residual_resources()
        
        # 3. 找到资源消耗最高的非约束节点作为起始节点
        if not non_constraint_indices:
            # 如果没有非约束节点，直接跳到约束节点放置
            start_node = None
            vn_placement_order = []
        else:
            max_demand_node = None
            max_demand = -1.0
            for vn_idx in non_constraint_indices:
                demand_info = self._get_vn_node_demand(vn, vn_idx)
                if demand_info['norm_demand'] > max_demand:
                    max_demand = demand_info['norm_demand']
                    max_demand_node = vn_idx
            
            # 4. 按广度优先遍历VN图，获取非约束节点的放置顺序
            vn_placement_order = self._bfs_vn_nodes(vn, max_demand_node, non_constraint_indices)
        
        # 5. 按BFS顺序贪心放置非约束节点
        non_constraint_mapping = {}
        temporary_deductions: List[tuple[int, float, float, float]] = []

        def _restore_temporary_deductions():
            if not temporary_deductions:
                return
            for sn_node, cpu, mem, disk in temporary_deductions:
                nd = self.env.G_sn.nodes[sn_node]
                nd['cpu_res'] += cpu
                nd['mem_res'] += mem
                nd['disk_res'] += disk
            temporary_deductions.clear()
        
        # 按BFS顺序放置非约束节点
        for vn_node in vn_placement_order:
            demand_info = self._get_vn_node_demand(vn, vn_node)
            demand_cpu = demand_info['abs_cpu']
            demand_mem = demand_info['abs_mem']
            demand_disk = demand_info['abs_disk']
            
            # 获取所有满足资源需求的SN节点，按noderank降序排序
            sn_nodes_with_rank = []
            for sn_node in self.env.G_sn.nodes:
                sn_idx = self.sn_node_to_idx.get(sn_node)
                if sn_idx is None or sn_idx >= len(sn_noderank):
                    continue
                
                nd = self.env.G_sn.nodes[sn_node]
                res_cpu = nd['cpu_res']
                res_mem = nd['mem_res']
                res_disk = nd['disk_res']
                
                # 检查是否满足需求
                if (res_cpu >= demand_cpu - 1e-9 and 
                    res_mem >= demand_mem - 1e-9 and 
                    res_disk >= demand_disk - 1e-9):
                    noderank = float(sn_noderank[sn_idx])
                    sn_nodes_with_rank.append({
                        'sn_node': sn_node,
                        'noderank': noderank,
                    })
            
            # 如果没有合适的SN节点，放置失败
            if not sn_nodes_with_rank:
                _restore_temporary_deductions()
                return False, {}, self.env.penalty
            
            # 按 noderank 降序排序，选择最高的
            sn_nodes_with_rank.sort(key=lambda x: x['noderank'], reverse=True)
            best_sn = sn_nodes_with_rank[0]['sn_node']
            non_constraint_mapping[vn_node] = best_sn

            # 立即在SN节点上扣减资源
            nd = self.env.G_sn.nodes[best_sn]
            nd['cpu_res'] -= demand_cpu
            nd['mem_res'] -= demand_mem
            nd['disk_res'] -= demand_disk
            temporary_deductions.append((best_sn, demand_cpu, demand_mem, demand_disk))
        
        # 6. 恢复临时扣减（因为后续会统一应用映射）
        _restore_temporary_deductions()
        
        # 4. 放置约束节点
        success, full_mapping, failure_reason = place_constraint_nodes(
            self.env, vn, non_constraint_mapping, constraint_mapping
        )
        
        if not success:
            return False, {}, self.env.penalty
        
        # 5. 验证映射并计算路径
        vn_paths = self.env._compute_paths_and_bw_demand(vn, full_mapping)
        if vn_paths is None:
            return False, {}, self.env.penalty

        # 6. 应用映射（扣减资源）
        self.env._apply_mapping(vn, full_mapping, vn_paths)
        
        # 7. 返回成功
        return True, full_mapping, 0.0  # r_t在外部计算


def run_gal_episode(
    sn_topology_path: str,
    workflow_types: Dict[str, str],
    device: str = 'cpu',
    arrival_rate: float = 0.05,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 1000,
    episode_seed: int = None,
    verbose: bool = True):
    """
    运行一个GAL episode（与fine-tuning完全相同的环境设置）
    
    Args:
        sn_topology_path: SN拓扑文件路径
        workflow_types: workflow类型字典
        device: 设备
        arrival_rate: 泊松到达率
        mean_lifetime: 平均生存时间
        max_arrived_tasks: 最大到达任务数
        max_time_steps: 最大时间步数
        episode_seed: episode随机种子
        verbose: 是否打印详细信息
    
    Returns:
        episode统计数据
    """
    
    # 构建环境
    env = SimuVNEEnv(
        sn_topology_path=sn_topology_path,
        device=device,
        penalty=-150.0,
        max_arrived_tasks=max_arrived_tasks
    )
    env.reset()
    
    # 获取SN容量用于VN特征归一化
    sn_capacity = env.get_sn_max_capacity()
    
    wf_gen = WorkflowGenerator(
        workflow_types=workflow_types,
        arrival_rate=arrival_rate,
        mean_lifetime=mean_lifetime,
        seed=episode_seed if episode_seed is not None else 42,
        sn_capacity_for_norm=sn_capacity
    )
    
    # 创建贪心放置器
    allocator = GreedyAllocator(env)
    
    # 时间驱动主循环
    time_step = 0
    placed_tasks = []
    
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
            
            if verbose:
                print(f"    [t={env.current_time:.1f}] 任务 #{task_id} 到达 "
                      f"(类型:{wf_type}, 节点数:{vn.x.size(0)}, 生存时间:{lifetime:.1f})", end='')
            
            # 贪心放置
            success, mapping, _ = allocator.greedy_place(vn)
            
            if success:
                # 加入存活集合
                vn_paths = env._compute_paths_and_bw_demand(vn, mapping)
                expire_time = env.current_time + lifetime
                env.active_workflows.append({
                    'vn': vn,
                    'mapping': mapping,
                    'paths': vn_paths,
                    'expire_time': expire_time,
                    'task_id': task_id,
                })
                env.accepted_count += 1
                
                # 计算r_t（跳数奖励）
                r_t = env._compute_rt()
                
                if verbose:
                    # 格式化映射关系：VN节点 -> SN节点
                    mapping_str = ', '.join([f"VN{vn_id}→SN{sn_id}" for vn_id, sn_id in sorted(mapping.items())])
                    print(f" → ✓成功 (r_t={r_t:.3f}, 存活任务数:{len(env.active_workflows)})")
                    print(f"       映射: {mapping_str}")
                
                placed_tasks.append({
                    'task_id': task_id,
                    'time': env.current_time,
                    'r_t': r_t,
                })
            else:
                r_t = env.penalty
                if verbose:
                    print(f" → ✗失败 (penalty={r_t:.3f})")
            
            # 记录轨迹
            env.traj.append({
                'time': env.current_time,
                'task_id': task_id,
                'success': success,
                'r_t': r_t,
                'done': False,
            })
        else:
            # 无任务到达，仍计算r_t
            r_t = env._compute_rt()
            env.traj.append({
                'time': env.current_time,
                'task_id': None,
                'success': None,
                'r_t': r_t,
                'done': False,
            })
        
        time_step += 1
    
    # 标记结束
    if len(env.traj) > 0:
        env.traj[-1]['done'] = True
    
    # 计算最终回报
    final_R = env.compute_final_return()
    
    if verbose:
        print(f"    Episode完成: 时间步={time_step}, 到达={env.arrived_count}, "
              f"接受={env.accepted_count}, 接受率={env.accepted_count/max(1,env.arrived_count)*100:.1f}%, "
              f"最终回报={final_R:.2f}")
    
    return {
        'final_return': final_R,
        'arrived': env.arrived_count,
        'accepted': env.accepted_count,
        'acceptance_rate': env.accepted_count / max(1, env.arrived_count),
        'time_steps': time_step,
        'placed_tasks': placed_tasks,
    }


def run_gal_benchmark(
    sn_topology_path: str,
    workflow_types: Dict[str, str],
    device: str = 'cpu',
    arrival_rate: float = 0.05,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 1000,
    num_episodes: int = 10,
    output_dir: str = '/home/yc2/mrt/a/gal_outputs'):
    """
    运行GAL基准测试
    
    Args:
        num_episodes: 运行的episode数量
        output_dir: 输出目录
    """
    print("=" * 80)
    print("GAL (Greedy Allocation Algorithm) - 基准测试")
    print("=" * 80)
    print(f"配置:")
    print(f"  到达率: {arrival_rate}")
    print(f"  平均生存时间: {mean_lifetime}")
    print(f"  每个episode最大任务数: {max_arrived_tasks}")
    print(f"  测试episode数: {num_episodes}")
    print("=" * 80)
    
    results = []
    start_time = time.time()
    
    for ep_idx in range(num_episodes):
        print(f"\n【Episode {ep_idx + 1}/{num_episodes}】")
        result = run_gal_episode(
            sn_topology_path=sn_topology_path,
            workflow_types=workflow_types,
            device=device,
            arrival_rate=arrival_rate,
            mean_lifetime=mean_lifetime,
            max_arrived_tasks=max_arrived_tasks,
            max_time_steps=max_time_steps,
            episode_seed=42 + ep_idx,
            verbose=True
        )
        results.append(result)
    
    total_time = time.time() - start_time
    
    # 统计结果
    print("\n" + "=" * 80)
    print("测试结果统计")
    print("=" * 80)
    
    avg_return = sum(r['final_return'] for r in results) / len(results)
    avg_acceptance = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_arrived = sum(r['arrived'] for r in results) / len(results)
    avg_accepted = sum(r['accepted'] for r in results) / len(results)
    
    print(f"平均最终回报: {avg_return:.2f}")
    print(f"平均接受率: {avg_acceptance:.2%}")
    print(f"平均到达任务数: {avg_arrived:.1f}")
    print(f"平均接受任务数: {avg_accepted:.1f}")
    print(f"总耗时: {total_time:.2f}秒")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f'gal_results_{timestamp}.json')
    
    summary = {
        'timestamp': timestamp,
        'config': {
            'arrival_rate': arrival_rate,
            'mean_lifetime': mean_lifetime,
            'max_arrived_tasks': max_arrived_tasks,
            'num_episodes': num_episodes,
        },
        'results': results,
        'summary': {
            'avg_return': avg_return,
            'avg_acceptance_rate': avg_acceptance,
            'avg_arrived': avg_arrived,
            'avg_accepted': avg_accepted,
            'total_time': total_time,
        }
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {result_file}")
    print("=" * 80)
    
    return summary


if __name__ == '__main__':
    # 配置
    sn_path = '/home/yc2/mrt/a/topo/SN_topology.json'
    workflow_types = {
        'workflow1': '/home/yc2/mrt/a/workflow_topo/workflow1_topo.json',
    }
    
    # 运行GAL基准测试
    summary = run_gal_benchmark(
        sn_topology_path=sn_path,
        workflow_types=workflow_types,
        device='cpu',
        arrival_rate=0.8,    # 与fine-tuning保持一致
        mean_lifetime=50.0,
        max_arrived_tasks=60,
        max_time_steps=2000,
        num_episodes=1,     # 运行10个episode进行统计
    )
    
    print("\n✓ GAL基准测试完成！")

