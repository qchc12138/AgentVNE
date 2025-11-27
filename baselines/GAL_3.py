#!/usr/bin/env python3
"""
GAL_3 (Greedy Allocation Algorithm - SN-Sorted) - 对比基准算法

贪心放置策略（SN节点预排序版本）：
1. 仅对SN节点做一次排序（按资源总量从高到低）
2. VN节点依次选择SN节点（从排序后的第一个开始，如果资源不够就选下一个）
3. 每次选择后立即扣减资源（影响后续VN节点的选择）

与GAL.py和GAL_2.py的区别：
- GAL.py: 对VN节点排序，为每个VN节点选择资源最多的SN节点
- GAL_2.py: 对VN节点排序，逐节点贪心选择并立即扣减资源
- GAL_3.py: 仅对SN节点排序一次，VN节点依次从排序后的SN节点中选择
"""

from typing import Dict

from torch_geometric.data import Data

from env import SimuVNEEnv


class GreedyAllocator:
    """贪心放置算法（SN节点预排序版本）"""
    
    def __init__(self, env: SimuVNEEnv):
        self.env = env
        self._sn_sorted_list = None  # 缓存的排序后的SN节点列表
    
    def _sort_sn_nodes(self) -> list:
        """
        对SN节点按资源总量从高到低排序（仅排序一次）。
        
        Returns:
            排序后的SN节点ID列表
        """
        if self._sn_sorted_list is not None:
            return self._sn_sorted_list
        
        sn_nodes = []
        for sn_node in self.env.G_sn.nodes:
            nd = self.env.G_sn.nodes[sn_node]
            res_cpu = nd['cpu_res']
            res_mem = nd['mem_res']
            res_disk = nd['disk_res']
            
            # 计算资源总量（归一化）
            norm_res_cpu = res_cpu / (self.env._sn_max_capacity['cpu_max'] + 1e-8)
            norm_res_mem = res_mem / (self.env._sn_max_capacity['mem_max'] + 1e-8)
            norm_res_disk = res_disk / (self.env._sn_max_capacity['disk_max'] + 1e-8)
            total_res = norm_res_cpu + norm_res_mem + norm_res_disk
            
            sn_nodes.append({
                'sn_node': sn_node,
                'total_res': total_res,
            })
        
        # 按资源总量从高到低排序
        sn_nodes.sort(key=lambda x: x['total_res'], reverse=True)
        self._sn_sorted_list = [item['sn_node'] for item in sn_nodes]
        
        return self._sn_sorted_list
    
    def _reset_sn_sort(self) -> None:
        """重置SN节点排序缓存（当资源发生变化时调用）。"""
        self._sn_sorted_list = None
    
    def greedy_place(self, vn: Data) -> tuple[bool, Dict[int, int], float]:
        """
        贪心放置策略（SN节点预排序版本）
        
        VN节点依次从排序后的SN节点列表中选择（从资源总量高的开始）。
        
        Args:
            vn: VN图数据（特征已归一化）
        
        Returns:
            (success, mapping, r_t): 成功标志、节点映射、奖励
        """
        N_v = vn.x.size(0)
        
        # 1. 重置排序缓存（因为资源状态可能已变化），然后获取排序后的SN节点列表（按资源总量从高到低）
        self._reset_sn_sort()
        sn_sorted_list = self._sort_sn_nodes()
        
        # 2. 依次为每个VN节点选择SN节点
        mapping = {}
        temporary_deductions = []  # 记录临时扣减的资源，用于失败时回滚
        
        for vn_idx in range(N_v):
            feats = vn.x[vn_idx]
            # 归一化需求
            norm_cpu = float(feats[0].item())
            norm_mem = float(feats[1].item())
            norm_disk = float(feats[2].item())
            
            # 转为绝对需求（用于资源检查）
            abs_cpu = norm_cpu * (self.env._sn_max_capacity['cpu_max'] + 1e-8)
            abs_mem = norm_mem * (self.env._sn_max_capacity['mem_max'] + 1e-8)
            abs_disk = norm_disk * (self.env._sn_max_capacity['disk_max'] + 1e-8)
            
            # 从排序后的SN节点列表中选择第一个满足资源需求的节点
            placed = False
            for sn_node in sn_sorted_list:
                nd = self.env.G_sn.nodes[sn_node]
                res_cpu = nd['cpu_res']  # 当前剩余资源（已反映之前的扣减）
                res_mem = nd['mem_res']
                res_disk = nd['disk_res']
                
                # 检查是否满足需求
                if (res_cpu >= abs_cpu - 1e-9 and 
                    res_mem >= abs_mem - 1e-9 and 
                    res_disk >= abs_disk - 1e-9):
                    # 找到合适的SN节点，进行映射
                    mapping[vn_idx] = sn_node
                    
                    # 立即扣减该SN节点的资源（影响后续VN节点的选择）
                    nd['cpu_res'] -= abs_cpu
                    nd['mem_res'] -= abs_mem
                    nd['disk_res'] -= abs_disk
                    
                    # 记录扣减信息（用于失败时回滚）
                    temporary_deductions.append((sn_node, abs_cpu, abs_mem, abs_disk))
                    
                    placed = True
                    break
            
            # 如果没有找到合适的SN节点，放置失败，回滚所有资源
            if not placed:
                # 回滚之前的资源扣减
                for sn_node, cpu, mem, disk in temporary_deductions:
                    nd = self.env.G_sn.nodes[sn_node]
                    nd['cpu_res'] += cpu
                    nd['mem_res'] += mem
                    nd['disk_res'] += disk
                return False, {}, self.env.penalty
        
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
        
        # 4. 先恢复临时扣减的资源，然后使用env的标准方法统一扣减资源（保持一致性）
        for sn_node, cpu, mem, disk in temporary_deductions:
            nd = self.env.G_sn.nodes[sn_node]
            nd['cpu_res'] += cpu
            nd['mem_res'] += mem
            nd['disk_res'] += disk
        
        # 使用env的标准方法统一扣减资源（保持一致性）
        self.env._apply_mapping(vn, mapping, vn_paths)
        
        # 5. 重置SN节点排序缓存（因为资源已发生变化）
        self._reset_sn_sort()
        
        # 6. 返回成功
        return True, mapping, 0.0  # r_t在外部计算


def run_gal3_episode(
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
    运行一个GAL_3 episode（与fine-tuning完全相同的环境设置）
    
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
    from env import SimuVNEEnv, WorkflowGenerator
    
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
            
            # 贪心放置（SN节点预排序版本）
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


if __name__ == '__main__':
    # 配置
    sn_path = '/home/yc2/mrt/AgentVNE/topo/SN_topology.json'
    workflow_types = {
        'workflow1': '/home/yc2/mrt/AgentVNE/workflow_topo/workflow1_topo.json',
    }
    
    # 运行GAL_3独立测试
    print("=" * 80)
    print("GAL_3 (SN-Sorted Greedy Allocation Algorithm) - 独立测试")
    print("=" * 80)
    
    result = run_gal3_episode(
        sn_topology_path=sn_path,
        workflow_types=workflow_types,
        device='cpu',
        arrival_rate=0.2,
        mean_lifetime=20.0,
        max_arrived_tasks=10,
        max_time_steps=100,
        episode_seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("测试结果:")
    print(f"  最终回报: {result['final_return']:.2f}")
    print(f"  接受率: {result['acceptance_rate']:.2%}")
    print(f"  到达任务数: {result['arrived']}")
    print(f"  接受任务数: {result['accepted']}")
    print("=" * 80)
    print("\n✓ GAL_3独立测试完成！")

