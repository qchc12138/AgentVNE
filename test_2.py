#!/usr/bin/env python3
"""
测试脚本：加载fine_tuning模型，生成任务并放置
优先级列表按概率值降序排列（不采样）
"""
import os
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from torch.distributions import Categorical
from torch_geometric.data import Data

from model import SimuVNE
from fine_tuning import ValueNet, PPOAgent
from env import SimuVNEEnv, WorkflowGenerator


class TestPPOAgent(PPOAgent):
    """测试版本的PPOAgent，优先级列表按概率值降序排列（不采样）"""
    
    def _generate_priority_lists(self, probs_matrix: torch.Tensor) -> List[List[int]]:
        """从概率矩阵生成优先级列表（按概率值降序排列，不采样）"""
        N_v, N_s = probs_matrix.shape
        priority_lists = []
        
        for i in range(N_v):
            probs = probs_matrix[i].cpu().numpy()  # [N_s]
            # 按概率值降序排序，得到SN节点索引
            sorted_indices = np.argsort(probs)[::-1]  # 降序
            priority_lists.append(sorted_indices.tolist())
        
        return priority_lists


def find_latest_finetuning_model(finetuning_dir='/home/zrz/SimuVNE/finetuning_putput'):
    """
    查找最新的fine-tuning模型
    
    Args:
        finetuning_dir: fine-tuning输出目录
    
    Returns:
        最新的模型路径，如果不存在则返回None
    """
    if not os.path.exists(finetuning_dir):
        return None
    
    import glob
    runs = sorted(glob.glob(f"{finetuning_dir}/run_*/policy_network.pth"))
    if runs:
        return runs[-1]  # 返回最新的模型
    return None


def test_placement_with_model(
    model_path: str,
    arrival_rate: float = 1.0,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 10,
    max_time_steps: int = 1000,
    seed: int = 42):
    """
    测试任务放置流程（时间驱动版本，优先级列表按概率值降序排列）
    
    Args:
        model_path: 模型checkpoint路径
        arrival_rate: 泊松到达率（每个时间单位的任务到达率）
        mean_lifetime: 平均生存时间（指数分布）
        max_arrived_tasks: 最大到达任务数
        max_time_steps: 最大时间步数
        seed: 随机种子
    """
    print("=" * 80)
    print("测试：加载fine-tuning模型并进行任务放置")
    print("优先级列表生成方式：按概率值降序排列（不采样）")
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
    
    # 加载模型
    print(f"\n【2. 加载模型】")
    policy = SimuVNE()
    value_net = ValueNet()
    
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
        raise
    
    # 创建测试版本的Agent（使用按概率值降序排列的优先级列表）
    agent = TestPPOAgent(policy, value_net, device=device)
    policy.eval()  # 设置为评估模式
    
    # 打印初始SN状态
    print(f"\n【3. 初始SN状态】")
    print("  【SN初始剩余容量】")
    for n in sorted(env.G_sn.nodes)[:10]:  # 只显示前10个节点
        nd = env.G_sn.nodes[n]
        print(f"    节点{n}: CPU={nd['cpu_res']:.2f}, Mem={nd['mem_res']:.2f}, Disk={nd['disk_res']:.2f}")
    if len(env.G_sn.nodes) > 10:
        print(f"    ... (共 {len(env.G_sn.nodes)} 个节点)")
    
    # 时间驱动主循环
    print(f"\n{'=' * 80}")
    print(f"【4. 开始任务放置】 (时间驱动: 到达率={arrival_rate}, 最大任务数={max_arrived_tasks}, 最大时间步={max_time_steps})")
    print(f"{'=' * 80}")
    
    placed_count = 0
    failed_count = 0
    time_step = 0
    r_t_list = []  # 记录每个任务放置后的 r_t 值
    last_task_arrival_time = None  # 记录最后一个任务到达的时间
    
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
            last_task_arrival_time = env.current_time  # 记录最后一个任务到达的时间
            
            # 打印任务到达信息
            print(f"\n{'-' * 80}")
            print(f"[t={env.current_time:.1f}] 任务 #{task_id} 到达 (类型:{wf_type}, 节点数:{vn.x.size(0)}, 生存时间:{lifetime:.1f})")
            print(f"{'-' * 80}")
            
            # 获取当前SN状态（包含剩余资源）
            sn_state = env.get_sn_state()
            
            # 调用策略网络生成放置方案（使用测试版本的Agent，优先级列表按概率值降序排列）
            mapping, logprob, value = agent.act(vn, sn_state, env=env, k_hop=1, verbose=True)
            
            # 检查是否成功放置（所有节点都已映射，资源已在act()中扣减）
            if len(mapping) == vn.x.size(0):
                # 所有节点都已放置，资源已在act()中扣减，只需要添加到存活集合
                vn_paths = env._compute_paths_and_bw_demand(vn, mapping)
                if vn_paths is None:
                    # 路径不存在，需要回滚资源
                    rollback_history = [(sn_id, vn_idx) for vn_idx, sn_id in mapping.items()]
                    agent._rollback_resource_deductions(env, rollback_history, vn, verbose=False)
                    success, r_t = False, env.penalty
                    print(f"  ✗ 放置失败: 路径不存在")
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
                    placed_count += 1
            else:
                # 部分节点未放置，资源已在act()中回滚
                success, r_t = False, env.penalty
                print(f"  ✗ 放置失败: 部分节点未放置 ({len(mapping)}/{vn.x.size(0)})")
                failed_count += 1
            
            # 记录 r_t 值（无论成功还是失败都记录）
            r_t_list.append(r_t)
            
            # 打印放置结果
            print(f"[t={env.current_time:.1f}] 任务 #{task_id} {'✓成功' if success else '✗失败'} (r_t={r_t:.3f}, 存活任务数:{len(env.active_workflows)})")
        
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
    
    # 计算到最后一个任务到来时的平均 r_t
    if len(r_t_list) > 0:
        avg_r_t = sum(r_t_list) / len(r_t_list)
        print(f"\n  【r_t 统计】")
        if last_task_arrival_time is not None:
            print(f"  最后一个任务到达时间: {last_task_arrival_time:.1f}")
        else:
            print(f"  最后一个任务到达时间: N/A")
        print(f"  到最后一个任务到来时的平均 r_t: {avg_r_t:.4f}")
        print(f"  r_t 值列表 (共{len(r_t_list)}个): {[f'{r:.3f}' for r in r_t_list]}")
        print(f"  r_t 最小值: {min(r_t_list):.4f}")
        print(f"  r_t 最大值: {max(r_t_list):.4f}")
        print(f"  r_t 标准差: {np.std(r_t_list):.4f}" if len(r_t_list) > 1 else "  r_t 标准差: N/A (仅1个值)")
    else:
        print(f"\n  【r_t 统计】")
        print(f"  无任务到达，无法计算 r_t")
    
    # 打印最终SN状态
    print(f"\n【6. 最终SN状态】")
    print("  【最终SN剩余容量】")
    for n in sorted(env.G_sn.nodes)[:10]:  # 只显示前10个节点
        nd = env.G_sn.nodes[n]
        print(f"    节点{n}: CPU={nd['cpu_res']:.2f}, Mem={nd['mem_res']:.2f}, Disk={nd['disk_res']:.2f}")
    if len(env.G_sn.nodes) > 10:
        print(f"    ... (共 {len(env.G_sn.nodes)} 个节点)")
    
    print(f"\n{'=' * 80}")
    print("测试完成！")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    # 查找最新的fine-tuning模型
    finetuning_dir = '/home/zrz/SimuVNE/finetuning_putput'
    latest_finetuning_model = find_latest_finetuning_model(finetuning_dir)
    
    if latest_finetuning_model:
        print("\n" + "=" * 80)
        print("【使用Fine-tuning模型进行测试】")
        print("=" * 80)
        print(f"模型路径: {latest_finetuning_model}")
        test_placement_with_model(
            model_path=latest_finetuning_model,
            arrival_rate=0.8,
            mean_lifetime=10.0,
            max_arrived_tasks=15,
            max_time_steps=1000,
            seed=42
        )
    else:
        print("\n" + "=" * 80)
        print("【错误】未找到fine-tuning模型")
        print("=" * 80)
        print(f"请确保在 {finetuning_dir} 目录下有训练好的模型")
        print("=" * 80)

