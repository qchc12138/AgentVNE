import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from model import SimuVNE
from env import SimuVNEEnv, WorkflowGenerator


class ValueNet(nn.Module):
    """价值网络：GNN编码SN与VN后做图级汇聚，输出V(s)标量。"""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.gcn1_v = GCNConv(input_dim, hidden_dim)
        self.gcn2_v = GCNConv(hidden_dim, hidden_dim)
        self.gcn1_s = GCNConv(input_dim, hidden_dim)
        self.gcn2_s = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, vn: Data, sn: Data) -> torch.Tensor:
        # 由于Data缺少batch，这里视为单图；global_mean_pool需batch张量
        x_v = F.relu(self.gcn1_v(vn.x, vn.edge_index))
        x_v = F.relu(self.gcn2_v(x_v, vn.edge_index))
        x_s = F.relu(self.gcn1_s(sn.x, sn.edge_index))
        x_s = F.relu(self.gcn2_s(x_s, sn.edge_index))
        b_v = torch.zeros(x_v.size(0), dtype=torch.long, device=x_v.device)
        b_s = torch.zeros(x_s.size(0), dtype=torch.long, device=x_s.device)
        gv = global_mean_pool(x_v, b_v)
        gs = global_mean_pool(x_s, b_s)
        g = torch.cat([gv, gs], dim=-1)
        v = self.mlp(g)
        return v.squeeze(-1)


class PPOAgent:
    def __init__(self,
                 policy: SimuVNE,
                 value_net: ValueNet,
                 lr_policy: float = 3e-4,
                 lr_value: float = 1e-3,
                 clip_ratio: float = 0.2,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 device: str = 'cpu'):
        self.policy = policy
        self.value_net = value_net
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.value_net.to(self.device)
        self.opt_pi = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.opt_v = optim.Adam(self.value_net.parameters(), lr=lr_value)

    @torch.no_grad()
    def act(self, vn: Data, sn: Data) -> Tuple[Dict[int, int], torch.Tensor, torch.Tensor]:
        # 输出放置概率矩阵 [N_v, N_s]，对VN每个节点按行采样SN节点
        vn = vn.to(self.device)
        sn = sn.to(self.device)
        probs_matrix = self.policy(vn, sn)  # softmax 已在模型内部做过
        N_v, N_s = probs_matrix.shape
        mapping: Dict[int, int] = {}
        logprob_sum = 0.0
        for i in range(N_v):
            probs = probs_matrix[i]
            cat = Categorical(probs=probs)
            a = cat.sample()
            mapping[i] = int(a.item())
            logprob_sum += float(cat.log_prob(a).item())
        value = self.value_net(vn, sn)
        return mapping, torch.tensor(logprob_sum, device=self.device, dtype=torch.float), value

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 按时间展开一次episode
        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float, device=self.device)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 0.0 if dones[t] else 1.0
            next_value = 0.0 if t == T - 1 else float(values[t + 1])
            delta = float(rewards[t]) + self.gamma * next_value * nonterminal - float(values[t])
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + torch.tensor(values, dtype=torch.float, device=self.device)
        # 归一化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def update(self,
               vn_list: List[Data],
               sn_list: List[Data],
               mappings: List[Dict[int, int]],
               logprobs_old: torch.Tensor,
               values_old: torch.Tensor,
               rewards: torch.Tensor,
               dones: List[bool],
               train_iters: int = 5):
        # 确保values_old是1维tensor
        if values_old.dim() > 1:
            values_old = values_old.squeeze()
        values_list = values_old.detach().cpu().tolist()
        # 确保values_list是一维列表
        if not isinstance(values_list, list):
            values_list = [values_list]
        elif len(values_list) > 0 and isinstance(values_list[0], list):
            # 如果是嵌套列表，展平
            values_list = [v[0] if isinstance(v, list) else v for v in values_list]
        
        adv, rets = self.compute_gae(rewards.tolist(), values_list, dones)

        for iter_idx in range(train_iters):
            # 策略更新
            print(f"      [PPO迭代 {iter_idx+1}/{train_iters}]", end=' ')
            new_logprobs = []
            entropies = []
            for vn, sn, mapping in zip(vn_list, sn_list, mappings):
                vn = vn.to(self.device)
                sn = sn.to(self.device)
                probs_matrix = self.policy(vn, sn)
                # 计算映射对应的logprob之和
                lp_sum = 0.0
                ent_sum = 0.0
                for i, j in mapping.items():
                    cat = Categorical(probs=probs_matrix[i])
                    lp_sum += cat.log_prob(torch.tensor(j, device=self.device)).sum()
                    ent_sum += cat.entropy().mean()
                new_logprobs.append(lp_sum)
                entropies.append(ent_sum)
            new_logprobs = torch.stack(new_logprobs)
            ent = torch.stack(entropies).mean()

            ratio = torch.exp(new_logprobs - logprobs_old)
            obj1 = ratio * adv
            obj2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            loss_pi = -(torch.min(obj1, obj2)).mean() - 0.001 * ent

            self.opt_pi.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.opt_pi.step()

            # 价值网络更新
            v_preds = []
            for vn, sn in zip(vn_list, sn_list):
                v_preds.append(self.value_net(vn.to(self.device), sn.to(self.device)))
            v_preds = torch.stack(v_preds)
            # 确保v_preds和rets形状一致
            if v_preds.dim() > 1:
                v_preds = v_preds.squeeze()
            if rets.dim() > 1:
                rets = rets.squeeze()
            loss_v = F.mse_loss(v_preds, rets)
            self.opt_v.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.opt_v.step()
            
            print(f"策略损失: {loss_pi.item():.4f}, 价值损失: {loss_v.item():.4f}")


def run_ppo_episode(
    agent: PPOAgent,
    sn_topology_path: str,
    workflow_types: Dict[str, str],
    device: str = 'cpu',
    arrival_rate: float = 0.05,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 1000,
    update_after_episode: bool = True,
    episode_seed: int = None):
    """
    运行一个PPO episode（时间驱动版本）：
    - 按时间单位推进
    - 泊松到达控制任务生成
    - 指数分布控制任务生存时间
    - 收集20个任务到达后结束
    
    Args:
        agent: PPO智能体（可在多个episode间共享）
        sn_topology_path: SN拓扑文件路径
        workflow_types: workflow类型字典
        device: 设备
        arrival_rate: 泊松到达率
        mean_lifetime: 平均生存时间
        max_arrived_tasks: 最大到达任务数
        max_time_steps: 最大时间步数
        update_after_episode: 如果True，episode结束后立即PPO更新；
                            如果False，只收集数据不更新（用于批量更新）
        episode_seed: episode随机种子（None则使用默认）
    
    Returns:
        episode统计数据 + 轨迹数据（如果update_after_episode=False）
    """

    # 构建环境与任务生成器
    env = SimuVNEEnv(
        sn_topology_path=sn_topology_path,
        device=device,
        penalty=-100.0,
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

    # 时间驱动主循环
    traj_vn = []
    traj_sn = []
    traj_map = []
    traj_logp = []
    traj_val = []
    traj_rew = []
    traj_done = []
    
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
            
            # 打印任务到达信息
            print(f"    [t={env.current_time:.1f}] 任务 #{task_id} 到达 (类型:{wf_type}, 节点数:{vn.x.size(0)}, 生存时间:{lifetime:.1f})", end='')
            
            # 获取当前SN状态（包含剩余资源）
            sn_state = env.get_sn_state()
            
            # 重试策略：最多尝试3次采样
            max_retries = 7
            success = False
            r_t = None
            mapping = None
            logprob = None
            value = None
            
            for attempt in range(1, max_retries + 1):
                # 调用策略网络生成放置方案
                mapping, logprob, value = agent.act(vn, sn_state)
                
                # 尝试放置
                success, r_t = env.try_place_task(vn, mapping, lifetime, task_id)
                
                if success:
                    # 成功放置，跳出重试循环
                    break
            
            # 打印放置结果
            status = "✓成功" if success else "✗失败"
            if success and attempt == 1:
                attempts_info = ""  # 第一次就成功，不显示尝试次数
            elif success and attempt > 1:
                attempts_info = f" (重试{attempt-1}次后成功)"  # 重试后成功
            else:
                attempts_info = f" (尝试{attempt}次均失败)"  # 所有尝试都失败
            print(f" → {status}{attempts_info} (r_t={r_t:.3f}, 存活任务数:{len(env.active_workflows)})")
            
            # 记录轨迹
            env.traj.append({
                'time': env.current_time,
                'task_id': task_id,
                'success': success,
                'r_t': r_t,
                'attempts': attempt,  # 记录尝试次数
                'done': False,
            })
            
            traj_vn.append(vn)
            traj_sn.append(sn_state)
            traj_map.append(mapping)
            traj_logp.append(logprob)
            traj_val.append(value)
            traj_rew.append(torch.tensor(r_t, dtype=torch.float, device=agent.device))
            traj_done.append(False)
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
            
            # 为了保持轨迹连续，使用零填充（可选）
            # 这里不添加到训练轨迹，仅记录在env.traj中
        
        time_step += 1
    
    # 标记结束
    if len(traj_done) > 0:
        traj_done[-1] = True
    if len(env.traj) > 0:
        env.traj[-1]['done'] = True
    
    # 计算最终回报
    final_R = env.compute_final_return()
    
    # 打印episode总结
    print(f"    Episode完成: 时间步={time_step}, 到达={env.arrived_count}, 接受={env.accepted_count}, 最终回报={final_R:.2f}")
    
    # PPO 更新（仅对有任务到达的时刻）
    if update_after_episode and len(traj_logp) > 0:
        logprobs_old = torch.stack(traj_logp)
        values_old = torch.stack(traj_val)
        rewards = torch.stack(traj_rew)
        agent.update(traj_vn, traj_sn, traj_map, logprobs_old, values_old, rewards, traj_done, train_iters=5)
    
    result = {
        'final_return': final_R,
        'arrived': env.arrived_count,
        'accepted': env.accepted_count,
        'traj_len': len(traj_rew),
        'time_steps': time_step,
    }
    
    # 如果用于批量更新，返回轨迹数据
    if not update_after_episode and len(traj_logp) > 0:
        result['trajectory'] = {
            'vn_list': traj_vn,
            'sn_list': traj_sn,
            'mappings': traj_map,
            'logprobs': torch.stack(traj_logp) if traj_logp else None,
            'values': torch.stack(traj_val) if traj_val else None,
            'rewards': torch.stack(traj_rew) if traj_rew else None,
            'dones': traj_done,
        }
    
    return result


def run_ppo_batch_training(
    sn_topology_path: str,
    workflow_types: Dict[str, str],
    policy_ckpt: str = None,
    device: str = 'cpu',
    arrival_rate: float = 0.05,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 1000,
    num_episodes_per_update: int = 4,
    train_iters: int = 5,
    num_updates: int = 10):
    """
    批量PPO训练：收集多个episode的数据后再更新
    
    Args:
        sn_topology_path: SN拓扑文件路径
        workflow_types: workflow类型字典
        policy_ckpt: 预训练模型路径（可选）
        device: 设备
        arrival_rate: 泊松到达率
        mean_lifetime: 平均生存时间
        max_arrived_tasks: 每个episode最大到达任务数
        max_time_steps: 每个episode最大时间步数
        num_episodes_per_update: 收集多少个episode后更新一次（批量大小）
        train_iters: 每次更新的迭代次数
        num_updates: 总共执行多少次批量更新
    
    Returns:
        training_stats: 训练统计信息列表
        agent: 训练后的PPOAgent对象
    """
    # 初始化策略和价值网络
    print(f"\n【初始化】创建策略网络和价值网络...")
    policy = SimuVNE()
    if policy_ckpt:
        ckpt = torch.load(policy_ckpt, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        policy.load_state_dict(state_dict, strict=False)
        print(f"  ✓ 加载预训练模型: {policy_ckpt}")
    else:
        print(f"  ✓ 使用随机初始化的策略网络")
    value_net = ValueNet()
    agent = PPOAgent(policy, value_net, device=device)
    print(f"  ✓ PPO Agent创建完成 (设备: {device})")
    
    training_stats = []
    
    for update_idx in range(num_updates):
        print(f"\n{'='*60}")
        print(f"批量更新 {update_idx + 1}/{num_updates}")
        print(f"{'='*60}")
        
        # 收集多个episode的轨迹数据
        all_vn_list = []
        all_sn_list = []
        all_mappings = []
        all_logprobs = []
        all_values = []
        all_rewards = []
        all_dones = []
        episode_stats = []
        
        for ep_idx in range(num_episodes_per_update):
            print(f"\n  【Episode {ep_idx + 1}/{num_episodes_per_update}】开始收集轨迹数据...")
            result = run_ppo_episode(
                agent=agent,
                sn_topology_path=sn_topology_path,
                workflow_types=workflow_types,
                device=device,
                arrival_rate=arrival_rate,
                mean_lifetime=mean_lifetime,
                max_arrived_tasks=max_arrived_tasks,
                max_time_steps=max_time_steps,
                update_after_episode=False,  # 不立即更新
                episode_seed=42 + update_idx * num_episodes_per_update + ep_idx
            )
            
            episode_stats.append({
                'final_return': result['final_return'],
                'arrived': result['arrived'],
                'accepted': result['accepted'],
            })
            
            # 累积轨迹数据
            if 'trajectory' in result:
                traj = result['trajectory']
                all_vn_list.extend(traj['vn_list'])
                all_sn_list.extend(traj['sn_list'])
                all_mappings.extend(traj['mappings'])
                all_logprobs.append(traj['logprobs'])
                all_values.append(traj['values'])
                all_rewards.append(traj['rewards'])
                all_dones.extend(traj['dones'])
            
            print(f"    → 完成 (接受率: {result['accepted']}/{result['arrived']}, 时间步:{result['time_steps']}, 回报:{result['final_return']:.2f})")
        
        # 批量PPO更新
        if len(all_logprobs) > 0:
            logprobs_old = torch.cat(all_logprobs, dim=0)
            values_old = torch.cat(all_values, dim=0)
            rewards = torch.cat(all_rewards, dim=0)
            
            print(f"\n  【PPO更新 {update_idx + 1}/{num_updates}】合并 {num_episodes_per_update} 个episode的数据...")
            print(f"    总样本数: {len(all_vn_list)}, 总奖励均值: {rewards.mean().item():.3f}")
            
            print(f"    开始PPO更新 (共{train_iters}次迭代):")
            agent.update(
                vn_list=all_vn_list,
                sn_list=all_sn_list,
                mappings=all_mappings,
                logprobs_old=logprobs_old,
                values_old=values_old,
                rewards=rewards,
                dones=all_dones,
                train_iters=train_iters
            )
            print(f"    PPO更新 {update_idx + 1}/{num_updates} 完成!")
            
            # 计算平均统计
            avg_return = sum(s['final_return'] for s in episode_stats) / len(episode_stats)
            avg_accepted = sum(s['accepted'] for s in episode_stats) / len(episode_stats)
            avg_arrived = sum(s['arrived'] for s in episode_stats) / len(episode_stats)
            
            print(f"  【更新 {update_idx + 1}/{num_updates} 结果】平均回报: {avg_return:.2f}, 平均接受率: {avg_accepted/avg_arrived:.2%} ({avg_accepted:.1f}/{avg_arrived:.1f})")
            
            training_stats.append({
                'update_idx': update_idx,
                'avg_return': avg_return,
                'avg_accepted': avg_accepted,
                'avg_arrived': avg_arrived,
                'total_samples': len(all_vn_list),
                'episode_stats': episode_stats,
            })
    
    return training_stats, agent


def save_training_results(training_stats: List[Dict], 
                          policy: SimuVNE, 
                          value_net: ValueNet,
                          output_dir: str = '/home/zrz/SimuVNE/finetuning_putput'):
    """
    保存训练结果、模型参数和可视化图表
    
    Args:
        training_stats: 训练统计信息列表
        policy: 策略网络
        value_net: 价值网络
        output_dir: 输出目录
    """
    # 创建输出目录（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"保存训练结果到: {run_dir}")
    print(f"{'='*60}")
    
    # 1. 提取训练数据
    updates = [s['update_idx'] + 1 for s in training_stats]
    avg_returns = [s['avg_return'] for s in training_stats]
    avg_acceptance_rates = [s['avg_accepted'] / s['avg_arrived'] for s in training_stats]
    total_samples = [s['total_samples'] for s in training_stats]
    
    # 2. 绘制训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO Training Results', fontsize=16, fontweight='bold')
    
    # 子图1: 平均回报变化
    axes[0, 0].plot(updates, avg_returns, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Update Number', fontsize=12)
    axes[0, 0].set_ylabel('Average Return', fontsize=12)
    axes[0, 0].set_title('Average Return per Update', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 子图2: 接受率变化
    axes[0, 1].plot(updates, avg_acceptance_rates, 'g-s', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Update Number', fontsize=12)
    axes[0, 1].set_ylabel('Acceptance Rate', fontsize=12)
    axes[0, 1].set_title('Task Acceptance Rate per Update', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% Target')
    axes[0, 1].legend()
    
    # 子图3: 样本数量
    axes[1, 0].bar(updates, total_samples, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Update Number', fontsize=12)
    axes[1, 0].set_ylabel('Total Samples', fontsize=12)
    axes[1, 0].set_title('Samples Collected per Update', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 子图4: 接受率和回报的综合对比
    ax4_1 = axes[1, 1]
    ax4_2 = ax4_1.twinx()
    
    line1 = ax4_1.plot(updates, avg_acceptance_rates, 'g-s', linewidth=2, markersize=6, label='Acceptance Rate')
    line2 = ax4_2.plot(updates, avg_returns, 'b-o', linewidth=2, markersize=6, label='Avg Return')
    
    ax4_1.set_xlabel('Update Number', fontsize=12)
    ax4_1.set_ylabel('Acceptance Rate', fontsize=12, color='g')
    ax4_2.set_ylabel('Average Return', fontsize=12, color='b')
    ax4_1.set_title('Acceptance Rate vs Return', fontsize=13, fontweight='bold')
    ax4_1.tick_params(axis='y', labelcolor='g')
    ax4_2.tick_params(axis='y', labelcolor='b')
    ax4_1.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4_1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(run_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线图已保存: {plot_path}")
    plt.close()
    
    # 3. 保存模型参数
    model_path = os.path.join(run_dir, 'policy_network.pth')
    torch.save({
        'model_state_dict': policy.state_dict(),
        'model_config': {
            'input_dim': policy.input_dim,
            'hidden_dim': policy.hidden_dim,
            'hist_dim': policy.hist_dim,
        }
    }, model_path)
    print(f"✓ 策略网络已保存: {model_path}")
    
    value_path = os.path.join(run_dir, 'value_network.pth')
    torch.save({
        'model_state_dict': value_net.state_dict(),
    }, value_path)
    print(f"✓ 价值网络已保存: {value_path}")
    
    # 4. 保存训练统计数据（JSON格式）
    stats_path = os.path.join(run_dir, 'training_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'num_updates': len(training_stats),
            'training_stats': training_stats,
            'summary': {
                'final_avg_return': avg_returns[-1],
                'final_acceptance_rate': avg_acceptance_rates[-1],
                'best_return': max(avg_returns),
                'best_acceptance_rate': max(avg_acceptance_rates),
                'total_samples': sum(total_samples),
            }
        }, f, indent=2)
    print(f"✓ 训练统计已保存: {stats_path}")
    
    # 5. 保存文本格式的训练摘要
    summary_path = os.path.join(run_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PPO Training Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Time: {timestamp}\n")
        f.write(f"Total Updates: {len(training_stats)}\n")
        f.write(f"Total Samples: {sum(total_samples)}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Training Progress:\n")
        f.write("-"*60 + "\n")
        for s in training_stats:
            f.write(f"Update {s['update_idx']+1}: "
                   f"Return={s['avg_return']:.2f}, "
                   f"Acceptance={s['avg_accepted']/s['avg_arrived']:.2%}, "
                   f"Samples={s['total_samples']}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Final Results:\n")
        f.write("-"*60 + "\n")
        f.write(f"Final Average Return: {avg_returns[-1]:.2f}\n")
        f.write(f"Final Acceptance Rate: {avg_acceptance_rates[-1]:.2%}\n")
        f.write(f"Best Return: {max(avg_returns):.2f}\n")
        f.write(f"Best Acceptance Rate: {max(avg_acceptance_rates):.2%}\n")
    
    print(f"✓ 训练摘要已保存: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"所有结果已成功保存到: {run_dir}")
    print(f"{'='*60}\n")
    
    return run_dir


if __name__ == '__main__':
    # 示例运行：使用仓库内示例拓扑（时间驱动版本）
    sn_path = '/home/zrz/SimuVNE/topo/SN_topology.json'
    workflow_types = {
        'workflow1': '/home/zrz/SimuVNE/workflow_topo/workflow1_topo.json',
        # 可扩展：'workflow2': '/path/to/workflow2_topo.json', ...
    }
    
    # ========== 方式1：单episode更新（每个episode结束后立即更新）==========
    # print("="*60)
    # print("方式1: 单episode更新（每个episode结束后立即更新）")
    # print("="*60)
    # policy1 = SimuVNE()
    # value_net1 = ValueNet()
    # agent1 = PPOAgent(policy1, value_net1, device='cpu')
    # 
    # stats1 = run_ppo_episode(
    #     agent=agent1,
    #     sn_topology_path=sn_path,
    #     workflow_types=workflow_types,
    #     device='cpu',
    #     arrival_rate=0.05,
    #     mean_lifetime=10.0,
    #     max_arrived_tasks=20,
    #     max_time_steps=1000,
    #     update_after_episode=True  # 立即更新
    # )
    # print('单episode更新统计:', stats1)
    
    # ========== 方式2：批量更新（收集多个episode后统一更新）==========
    print("\n" + "="*60)
    print("方式2: 批量更新（收集4个episode后统一更新）")
    print("="*60)
    
    training_stats, agent = run_ppo_batch_training(
        sn_topology_path=sn_path,
        workflow_types=workflow_types,
        policy_ckpt=None,  # 旧代码：使用随机初始化
        # policy_ckpt='/home/zrz/SimuVNE/pretrain_outputs/checkpoint_best.pt',  # 使用预训练最优模型
        device='cpu',
        arrival_rate=0.5,   # arrival_rate = 0.2 表示每5个时间单位到达1个任务
        mean_lifetime=8.0,
        max_arrived_tasks=30,
        max_time_steps=2000,
        num_episodes_per_update=4,  # 收集4个episode后更新一次
        train_iters=3,  # 每次更新PPO算法迭代3次
        num_updates=5  # 总共30次批量更新
    )
    
    print("\n批量训练统计:")
    for stat in training_stats:
        print(f"  更新 {stat['update_idx']+1}: 平均回报={stat['avg_return']:.2f}, "
              f"接受率={stat['avg_accepted']/stat['avg_arrived']:.2%}, "
              f"样本数={stat['total_samples']}")
    
    # 保存训练结果、模型参数和可视化图表
    save_training_results(
        training_stats=training_stats,
        policy=agent.policy,
        value_net=agent.value_net,
        output_dir='/home/zrz/SimuVNE/finetuning_putput'
    )


