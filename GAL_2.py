#!/usr/bin/env python3
"""
GAL_2 (Greedy Allocation Algorithm - Per-Node) - 对比基准算法

贪心放置策略（逐节点版本）：
1. 按VN节点的归一化需求强度(cpu+mem+disk)从大到小排序
2. 依次为每个VN节点贪心选择SN节点
3. 每次选择后立即扣减资源（影响后续VN节点的选择）

与GAL.py的区别：
- GAL.py: 一次性为所有VN节点做贪心决策，然后统一扣减资源
- GAL_2.py: 逐个VN节点进行贪心决策和资源扣减（更贪心）
"""

import json
import time
from typing import Dict, List
from datetime import datetime
import os

import torch
from torch_geometric.data import Data

from env import SimuVNEEnv, WorkflowGenerator


class GreedyAllocator:
    """贪心放置算法（逐节点版本）"""
    
    def __init__(self, env: SimuVNEEnv):
        self.env = env
    
    def greedy_place(self, vn: Data) -> tuple[bool, Dict[int, int], float]:
        """
        贪心放置策略（逐节点版本）
        
        每个VN节点独立进行贪心选择，选择后立即扣减资源
        
        Args:
            vn: VN图数据（特征已归一化）
        
        Returns:
            (success, mapping, r_t): 成功标志、节点映射、奖励
        """
        N_v = vn.x.size(0)
        
        # 1. 计算VN节点的归一化需求强度（cpu+mem+disk）
        vn_demands = []
        for i in range(N_v):
            feats = vn.x[i]
            # 归一化需求
            norm_cpu = float(feats[0].item())
            norm_mem = float(feats[1].item())
            norm_disk = float(feats[2].item())
            norm_demand = norm_cpu + norm_mem + norm_disk
            
            # 转为绝对需求（用于资源检查）
            abs_cpu = norm_cpu * (self.env._sn_max_capacity['cpu_max'] + 1e-8)
            abs_mem = norm_mem * (self.env._sn_max_capacity['mem_max'] + 1e-8)
            abs_disk = norm_disk * (self.env._sn_max_capacity['disk_max'] + 1e-8)
            
            vn_demands.append({
                'vn_node': i,
                'norm_demand': norm_demand,
                'abs_cpu': abs_cpu,
                'abs_mem': abs_mem,
                'abs_disk': abs_disk,
            })
        
        # 按归一化需求从大到小排序
        vn_demands.sort(key=lambda x: x['norm_demand'], reverse=True)
        
        # 2. 逐节点贪心映射（每次选择后立即扣减资源）
        mapping = {}
        temporary_deductions = []  # 记录临时扣减的资源，用于失败时回滚
        
        for vn_info in vn_demands:
            vn_node = vn_info['vn_node']
            demand_cpu = vn_info['abs_cpu']
            demand_mem = vn_info['abs_mem']
            demand_disk = vn_info['abs_disk']
            
            # 计算所有SN节点的剩余资源强度（使用当前实时资源状态）
            sn_candidates = []
            for sn_node in self.env.G_sn.nodes:
                nd = self.env.G_sn.nodes[sn_node]
                res_cpu = nd['cpu_res']  # 当前剩余资源（已反映之前的扣减）
                res_mem = nd['mem_res']
                res_disk = nd['disk_res']
                
                # 检查是否满足需求
                if (res_cpu >= demand_cpu - 1e-9 and 
                    res_mem >= demand_mem - 1e-9 and 
                    res_disk >= demand_disk - 1e-9):
                    # 计算剩余资源强度（归一化）
                    norm_res_cpu = res_cpu / (self.env._sn_max_capacity['cpu_max'] + 1e-8)
                    norm_res_mem = res_mem / (self.env._sn_max_capacity['mem_max'] + 1e-8)
                    norm_res_disk = res_disk / (self.env._sn_max_capacity['disk_max'] + 1e-8)
                    res_strength = norm_res_cpu + norm_res_mem + norm_res_disk
                    
                    sn_candidates.append({
                        'sn_node': sn_node,
                        'res_strength': res_strength,
                    })
            
            # 如果没有合适的SN节点，放置失败，回滚所有资源
            if not sn_candidates:
                # 回滚之前的资源扣减
                for sn_node, cpu, mem, disk in temporary_deductions:
                    nd = self.env.G_sn.nodes[sn_node]
                    nd['cpu_res'] += cpu
                    nd['mem_res'] += mem
                    nd['disk_res'] += disk
                return False, {}, self.env.penalty
            
            # 选择剩余资源最多的SN节点
            sn_candidates.sort(key=lambda x: x['res_strength'], reverse=True)
            best_sn = sn_candidates[0]['sn_node']
            mapping[vn_node] = best_sn
            
            # 立即扣减该SN节点的资源（影响后续VN节点的选择）
            nd = self.env.G_sn.nodes[best_sn]
            nd['cpu_res'] -= demand_cpu
            nd['mem_res'] -= demand_mem
            nd['disk_res'] -= demand_disk
            
            # 记录扣减信息（用于失败时回滚）
            temporary_deductions.append((best_sn, demand_cpu, demand_mem, demand_disk))
        
        # 3. 验证映射并计算路径
        vn_paths = self.env._compute_paths_and_bw_demand(vn, mapping)
        if vn_paths is None:
            # 回滚所有资源扣减
            for sn_node, cpu, mem, disk in temporary_deductions:
                nd = self.env.G_sn.nodes[sn_node]
                nd['cpu_res'] += cpu
                nd['mem_res'] += mem
                nd['disk_res'] += disk
            return False, {}, self.env.penalty
        
        # 4. 注意：资源已经在循环中扣减了，这里不需要再调用_apply_mapping
        # 但我们仍需要处理带宽扣减（如果需要的话）
        # 由于env._apply_mapping会再次扣减节点资源，我们需要先恢复，再统一扣减
        
        # 先恢复临时扣减的资源
        for sn_node, cpu, mem, disk in temporary_deductions:
            nd = self.env.G_sn.nodes[sn_node]
            nd['cpu_res'] += cpu
            nd['mem_res'] += mem
            nd['disk_res'] += disk
        
        # 使用env的标准方法统一扣减资源（保持一致性）
        self.env._apply_mapping(vn, mapping, vn_paths)
        
        # 5. 返回成功
        return True, mapping, 0.0  # r_t在外部计算


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
            
            # 贪心放置（逐节点版本）
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
    print("GAL_2 (Greedy Allocation Algorithm - Per-Node) - 基准测试")
    print("=" * 80)
    print(f"配置:")
    print(f"  到达率: {arrival_rate}")
    print(f"  平均生存时间: {mean_lifetime}")
    print(f"  每个episode最大任务数: {max_arrived_tasks}")
    print(f"  测试episode数: {num_episodes}")
    print(f"  策略: 逐节点贪心（每个VN节点选择后立即扣减资源）")
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
    result_file = os.path.join(output_dir, f'gal2_results_{timestamp}.json')
    
    summary = {
        'timestamp': timestamp,
        'algorithm': 'GAL_2 (Per-Node Greedy)',
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
    
    # 运行GAL_2基准测试
    summary = run_gal_benchmark(
        sn_topology_path=sn_path,
        workflow_types=workflow_types,
        device='cpu',
        arrival_rate=0.8,    # 与fine-tuning保持一致
        mean_lifetime=50.0,
        max_arrived_tasks=60,
        max_time_steps=2000,
        num_episodes=1,     # 运行1个episode进行测试
    )
    
    print("\n✓ GAL_2基准测试完成！")
