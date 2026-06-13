"""
    PPO fine-tuning script
    Will be released after the paper is accepted
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional

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
import networkx as nx

from model import SimuVNE
from env import SimuVNEEnv, WorkflowGenerator

import random
import numpy as np
import sys

class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 在main函数开头调用

class ValueNet(nn.Module):
    """Value network: estimates V(sn_state, vn_state) as a scalar.

    Two-branch GCN encoder (SN + VN) -> global mean pool -> MLP head.
    """

    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sn_conv1 = GCNConv(input_dim, hidden_dim)
        self.sn_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.vn_conv1 = GCNConv(input_dim, hidden_dim)
        self.vn_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, x):
        return x.mean(dim=0, keepdim=True)   # [1, hidden_dim]

    def forward(self, sn_data, vn_data):
        """Returns scalar state value.

        Args:
            sn_data: PyG Data for substrate network
            vn_data: PyG Data for virtual network (or None)
        """
        # --- SN branch ---
        h_sn = F.relu(self.sn_conv1(sn_data.x, sn_data.edge_index))
        h_sn = F.relu(self.sn_conv2(h_sn, sn_data.edge_index))
        sn_vec = self._pool(h_sn)

        # --- VN branch ---
        if vn_data is not None and vn_data.x.size(0) > 0:
            h_vn = F.relu(self.vn_conv1(vn_data.x, vn_data.edge_index))
            h_vn = F.relu(self.vn_conv2(h_vn, vn_data.edge_index))
            vn_vec = self._pool(h_vn)
        else:
            vn_vec = torch.zeros(1, self.hidden_dim, device=sn_vec.device)

        combined = torch.cat([sn_vec, vn_vec], dim=-1)   # [1, 2*hidden_dim]
        return self.fc(combined).squeeze()              # scalar


class PPOAgent:
    """PPO agent holding a SimuVNE policy and a ValueNet critic."""

    def __init__(
        self, policy, value_net, device='cpu',
        lr_policy=1e-4, lr_value=1e-3,
        gamma=0.99, gae_lambda=0.95,
        clip_epsilon=0.2, entropy_coef=0.01,
        value_loss_coef=0.5, max_grad_norm=0.5,
        train_iters=5,
    ):
        self.policy = policy.to(device)
        self.value_net = value_net.to(device)
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.train_iters = train_iters

        self.optimizer_policy = optim.Adam(policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(value_net.parameters(), lr=lr_value)

        self.trajectories = []   # list of episode trajectory dicts

    # ---- trajectory helpers ------------------------------------------

    def compute_log_prob(self, action_probs, mapping):
        """Joint log-prob of a greedy placement *mapping* given *action_probs*."""
        lp = 0.0
        for vn_idx, sn_idx in mapping.items():
            lp = lp + torch.log(action_probs[vn_idx, sn_idx] + 1e-8)
        return lp

    def compute_gae(self, rewards, values, dones):
        """Generalised Advantage Estimation.

        Returns (returns, advantages) as lists of floats.
        """
        T = len(rewards)
        returns = [0.0] * T
        advantages = [0.0] * T
        gae = 0.0
        next_val = 0.0
        for t in reversed(range(T)):
            non_term = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * non_term - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * non_term
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_val = values[t]
        return returns, advantages

    # ---- storage -----------------------------------------------------

    def store_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    # ---- PPO update --------------------------------------------------

    def update(self):
        """Multi-epoch PPO-Clip update on all stored trajectories."""
        if not self.trajectories:
            return {'policy_loss': 0.0, 'value_loss': 0.0}

        # Flatten all steps
        all_sn, all_vn = [], []
        all_log_probs, all_returns, all_advantages = [], [], []
        all_mappings = []

        for traj in self.trajectories:
            steps = traj['steps']
            returns, advs = self.compute_gae(
                [s['reward'] for s in steps],
                [s['value'] for s in steps],
                [s['done'] for s in steps],
            )
            for i, s in enumerate(steps):
                all_sn.append(s['sn_data'])
                all_vn.append(s['vn_data'])
                all_log_probs.append(s['log_prob'])
                all_mappings.append(s['mapping'])
            all_returns.extend(returns)
            all_advantages.extend(advs)

        N = len(all_sn)
        if N == 0:
            self.trajectories = []
            return {'policy_loss': 0.0, 'value_loss': 0.0}

        adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(all_log_probs, dtype=torch.float32, device=self.device)
        # Standardize returns for stable value network training
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
        ret_t = torch.where(ret_t.isnan(), torch.zeros_like(ret_t), ret_t)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        adv_t = torch.where(adv_t.isnan(), torch.zeros_like(adv_t), adv_t)

        for _ in range(self.train_iters):
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()

            total_p_loss, total_v_loss = 0.0, 0.0
            total_entropy = 0.0
            for i in range(N):
                sn_data = all_sn[i].to(self.device)
                vn_data = all_vn[i].to(self.device) if all_vn[i] is not None else None

                action_probs_raw = self.policy(vn_data, sn_data)   # [N_vn, N_sn]
                action_probs = action_probs_raw                       # [N_vn, N_sn]
                # Row-normalize: each VN row sums to 1 (valid distribution over SN)
                action_probs = action_probs / (action_probs.sum(dim=1, keepdim=True) + 1e-8)

                # Per-sample entropy to encourage exploration
                ent = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
                total_entropy += ent

                new_lp = self.compute_log_prob(action_probs, all_mappings[i])  # expects [N_vn, N_sn]
                new_val = self.value_net(sn_data, vn_data)

                ratio = torch.exp(new_lp - old_lp[i])
                surr1 = ratio * adv_t[i]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * adv_t[i]
                p_loss = -torch.min(surr1, surr2)
                v_loss = F.mse_loss(new_val, ret_t[i])

                total_p_loss += p_loss
                total_v_loss += v_loss

            avg_p_loss = total_p_loss / N
            avg_v_loss = total_v_loss / N
            avg_entropy = total_entropy / N
            loss = avg_p_loss + self.value_loss_coef * avg_v_loss - self.entropy_coef * avg_entropy

            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.optimizer_policy.step()
            self.optimizer_value.step()

        self.trajectories = []
        return {'policy_loss': avg_p_loss.item(), 'value_loss': avg_v_loss.item()}


def _build_env(sn_topo_path, workflow_types, arrival_rate, mean_lifetime,
               max_arrived_tasks, max_time_steps, seed):
    """Create WorkflowGenerator + SimuVNEEnv pair."""
    wf_gen = WorkflowGenerator(
        arrival_rate=arrival_rate,
        mean_lifetime=mean_lifetime,
        workflow_types=workflow_types,
        max_arrived_tasks=max_arrived_tasks,
        seed=seed,
    )
    env = SimuVNEEnv(
        sn_topology_path=sn_topo_path,
        workflow_generator=wf_gen,
        max_time_steps=max_time_steps,
        seed=seed,
    )
    return env


def run_ppo_episode(
    agent,
    sn_topology_path,
    workflow_types,
    device='cpu',
    arrival_rate=0.05,
    mean_lifetime=10.0,
    max_arrived_tasks=20,
    max_time_steps=1000,
    update_after_episode=True,
    episode_seed=None,
    verbose=False,
):
    """Run a single PPO episode, collect trajectory, optionally update.

    Returns:
        dict with keys: total_return, num_steps, accepted, arrived, avg_rt,
                        trajectory
    """
    env = _build_env(sn_topology_path, workflow_types, arrival_rate,
                     mean_lifetime, max_arrived_tasks, max_time_steps,
                     seed=episode_seed)

    (sn_data, vn_data) = env.reset()
    done = False
    episode_return = 0.0
    steps = []

    while not done:
        if vn_data is None:
            # no VN pending -- advance time without policy forward
            (sn_data, vn_data), _, done, _ = env.step(
                torch.zeros(1, env.num_sn_nodes))
            continue

        sn_dev = sn_data.to(device)
        vn_dev = vn_data.to(device)

        with torch.no_grad():
            action_probs_raw = agent.policy(vn_dev, sn_dev)       # [N_vn, N_sn]
            action_probs = action_probs_raw                       # already [N_vn, N_sn]
            # Row-normalize: each VN row sums to 1 (valid distribution over SN)
            action_probs = action_probs / (action_probs.sum(dim=1, keepdim=True) + 1e-8)
            value = agent.value_net(sn_dev, vn_dev)

        (next_sn, next_vn), reward, done, info = env.step(action_probs)

        mapping = info.get('mapping', {})
        log_prob = (agent.compute_log_prob(action_probs, mapping)  # expects [N_vn, N_sn]
                    if mapping else 0.0)

        steps.append({
            'sn_data': sn_data,
            'vn_data': vn_data,
            'log_prob': log_prob,
            'value': value.item(),
            'reward': reward,
            'done': done,
            'mapping': mapping,
            'info': info,
        })

        episode_return += reward
        sn_data, vn_data = next_sn, next_vn

    stats = env.get_stats()

    trajectory = {
        'steps': steps,
        'total_return': episode_return,
        'stats': stats,
    }

    if update_after_episode:
        agent.store_trajectory(trajectory)
        agent.update()

    return {
        'total_return': episode_return,
        'num_steps': len(steps),
        'accepted': stats['total_accepted'],
        'arrived': stats['total_arrived'],
        'avg_rt': episode_return / max(len(steps), 1),
        'trajectory': trajectory,
    }


def run_ppo_batch_training(
    sn_topology_path,
    workflow_types,
    policy_ckpt=None,
    value_ckpt=None,
    device='cpu',
    arrival_rate=0.05,
    mean_lifetime=10.0,
    max_arrived_tasks=20,
    max_time_steps=1000,
    num_episodes_per_update=4,
    train_iters=5,
    num_updates=10,
    verbose=False,
):
    """Batch PPO training loop.

    Returns:
        (training_stats, agent)
    """
    # ---- resolve VN node count from first workflow template ----
    import json as _json
    first_wf = next(iter(workflow_types.values()))
    with open(first_wf, 'r', encoding='utf-8') as f:
        wf_topo = _json.load(f)
    num_nodes_j = len(json.load(open(sn_topology_path, 'r', encoding='utf-8'))['nodes'])

    # ---- build models ----
    policy = SimuVNE(input_dim=6, hidden_dim=64, hist_dim=32,
                     num_nodes_j=num_nodes_j)
    value_net = ValueNet(input_dim=6, hidden_dim=64)

    if policy_ckpt is not None and os.path.exists(policy_ckpt):
        ckpt = torch.load(policy_ckpt, map_location=device)
        if 'model_state_dict' in ckpt:
            policy.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded pretrained policy: {policy_ckpt}")
        else:
            policy.load_state_dict(ckpt)
            print(f"  Loaded pretrained policy: {policy_ckpt}")
    else:
        print("  Using randomly initialized policy network")

    if value_ckpt is not None and os.path.exists(value_ckpt):
        ckpt_v = torch.load(value_ckpt, map_location=device)
        value_net.load_state_dict(ckpt_v['model_state_dict'])
        print(f"  Loaded pretrained value net: {value_ckpt}")
    else:
        print("  Using randomly initialized value network")

    agent = PPOAgent(policy, value_net, device=device,
                     train_iters=train_iters)
    print(f"  PPO Agent created (device: {device})")

    training_stats = []

    for update_idx in range(num_updates):
        ep_returns, ep_accepted, ep_arrived = [], [], []
        total_steps = 0

        for ep in range(num_episodes_per_update):
            ep_seed = (random.randint(0, 2**31 - 1)
                       if num_episodes_per_update > 1 else None)
            ep_result = run_ppo_episode(
                agent=agent,
                sn_topology_path=sn_topology_path,
                workflow_types=workflow_types,
                device=device,
                arrival_rate=arrival_rate,
                mean_lifetime=mean_lifetime,
                max_arrived_tasks=max_arrived_tasks,
                max_time_steps=max_time_steps,
                update_after_episode=False,
                episode_seed=ep_seed,
                verbose=verbose,
            )
            ep_returns.append(ep_result['total_return'])
            ep_accepted.append(ep_result['accepted'])
            ep_arrived.append(ep_result['arrived'])
            total_steps += ep_result['num_steps']

            # store for batch update later
            agent.store_trajectory(ep_result['trajectory'])

        # ---- batch PPO update ----
        update_info = agent.update()

        avg_return = sum(ep_returns) / len(ep_returns)
        total_accepted = sum(ep_accepted)
        total_arrived = sum(ep_arrived)

        if verbose:
            print(f"  [PPO update {update_idx + 1}/{num_updates}] "
                  f"policy_loss={update_info['policy_loss']:.4f}  "
                  f"value_loss={update_info['value_loss']:.4f}  "
                  f"avg_return={avg_return:.2f}  "
                  f"acceptance={total_accepted}/{total_arrived}")

        training_stats.append({
            'update_idx': update_idx,
            'avg_return': avg_return,
            'avg_accepted': total_accepted,
            'avg_arrived': total_arrived,
            'total_samples': total_steps,
            'policy_loss': update_info['policy_loss'],
            'value_loss': update_info['value_loss'],
        })

    return training_stats, agent


def save_training_results(training_stats: List[Dict], 
                          policy: SimuVNE, 
                          value_net: ValueNet,
                          output_dir: str = None,
                          run_dir: str = None):
    """
    保存训练结果、模型参数和可视化图表

    """
    # 如果指定了run_dir，直接使用
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        # 从run_dir提取output_dir（用于保存latest模型）
        output_dir = os.path.dirname(run_dir)
        # 从run_dir提取timestamp（用于保存统计信息）
        timestamp = os.path.basename(run_dir).replace('run_', '')
    else:
        # 如果没有指定输出目录，使用默认的相对路径
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'finetuning_output_4')
        
        # 转换为绝对路径
        if not os.path.isabs(output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, output_dir)
        
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
    
    # 提取每个episode的平均r_t（所有时间步的r_t的平均值，用于打印）
    all_episode_avg_rt = []
    for s in training_stats:
        if 'episode_stats' in s:
            for ep_stat in s['episode_stats']:
                all_episode_avg_rt.append(ep_stat['avg_rt'])
    
    # 打印每个episode的平均r_t
    if all_episode_avg_rt:
        print(f"\n每个episode的平均r_t（所有时间步r_t的均值）:")
        for idx, avg_rt in enumerate(all_episode_avg_rt, 1):
            print(f"  Episode {idx}: {avg_rt:.3f}")
        print(f"  平均: {sum(all_episode_avg_rt) / len(all_episode_avg_rt):.3f}")
    
    # 2. 绘制训练曲线（三个独立的图）
    
    # 图1: Average total return per episode
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(updates, avg_returns, 'b-o', linewidth=2, markersize=1)
    plt.xlabel('Update Number', fontsize=12)
    plt.ylabel('Average Total Return per Episode', fontsize=12)
    plt.title('Average Total Return per Episode', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path1 = os.path.join(run_dir, 'training_curves_return.png')
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    print(f"? 训练曲线图（总回报）已保存: {plot_path1}")
    plt.close()
    
    # 图2: Acceptance rate
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(updates, avg_acceptance_rates, 'g-s', linewidth=2, markersize=1)
    plt.xlabel('Update Number', fontsize=12)
    plt.ylabel('Acceptance Rate', fontsize=12)
    plt.title('Task Acceptance Rate per Update', fontsize=13, fontweight='bold')
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% Target')
    plt.legend()
    plt.tight_layout()
    plot_path2 = os.path.join(run_dir, 'training_curves_acceptance.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    print(f"? 训练曲线图（接受率）已保存: {plot_path2}")
    plt.close()
    
    # 图3: Average r_t per episode (mean of all time steps)
    episode_indices = list(range(1, len(all_episode_avg_rt) + 1))
    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(episode_indices, all_episode_avg_rt, 'r-^', linewidth=2, markersize=1)
    plt.xlabel('Episode Number', fontsize=12)
    plt.ylabel('Average r_t per Episode', fontsize=12)
    plt.title('Average r_t per Episode (Mean of All Time Steps)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path3 = os.path.join(run_dir, 'training_curves_rt.png')
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    print(f"? 训练曲线图（平均r_t）已保存: {plot_path3}")
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
    print(f"? 策略网络已保存: {model_path}")
    
    value_path = os.path.join(run_dir, 'value_network.pth')
    torch.save({
        'model_state_dict': value_net.state_dict(),
    }, value_path)
    print(f"? 价值网络已保存: {value_path}")
    
    # 3.5. 额外保存最新模型到 finetuning_output_4 目录（不带时间戳）
    latest_policy_path = os.path.join(output_dir, 'policy_network_latest.pth')
    torch.save({
        'model_state_dict': policy.state_dict(),
        'model_config': {
            'input_dim': policy.input_dim,
            'hidden_dim': policy.hidden_dim,
            'hist_dim': policy.hist_dim,
        }
    }, latest_policy_path)
    print(f"? 策略网络（最新）已保存: {latest_policy_path}")
    
    latest_value_path = os.path.join(output_dir, 'value_network_latest.pth')
    torch.save({
        'model_state_dict': value_net.state_dict(),
    }, latest_value_path)
    print(f"? 价值网络（最新）已保存: {latest_value_path}")
    
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
    print(f"? 训练统计已保存: {stats_path}")
    
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
    
    print(f"? 训练摘要已保存: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"所有结果已成功保存到: {run_dir}")
    print(f"{'='*60}\n")
    
    return run_dir


def get_model_paths(script_dir: str, use_finetuning_model: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    根据 use_finetuning_model 参数获取模型路径

    """
    if use_finetuning_model:
        # 使用 finetuning_output_4 目录下的最新模型（微调后的模型）
        finetuning_policy_path = os.path.join(script_dir, 'finetuning_output', 'policy_network_latest.pth')
        finetuning_value_path = os.path.join(script_dir, 'finetuning_output', 'value_network_latest.pth')
        if os.path.exists(finetuning_policy_path):
            policy_ckpt_path = finetuning_policy_path
            value_ckpt_path = finetuning_value_path if os.path.exists(finetuning_value_path) else None
            print(f"  ? 使用微调后的最新模型: {policy_ckpt_path}")
            return policy_ckpt_path, value_ckpt_path
        else:
            print(f"  ??  微调模型不存在，使用随机初始化")
            return None, None
    else:
        # 使用 pretrain_outputs 目录下的预训练模型
        pretrain_policy_path = os.path.join(script_dir, 'pretrain_outputs', 'checkpoint_latest.pt')
        if os.path.exists(pretrain_policy_path):
            policy_ckpt_path = pretrain_policy_path
            value_ckpt_path = None  # pretrain_outputs目录下没有单独的价值网络
            print(f"  ? 使用预训练模型: {policy_ckpt_path}")
            return policy_ckpt_path, value_ckpt_path
        else:
            print(f"  ??  预训练模型不存在，使用随机初始化")
            return None, None


if __name__ == '__main__':
    # 获取脚本所在目录，用于构建相对路径的默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建运行目录和日志文件（保存所有打印输出）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(script_dir, 'finetuning_output_4')
    os.makedirs(output_base_dir, exist_ok=True)
    run_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, 'log.txt')
    
    # 打开日志文件并设置Tee，同时输出到控制台和文件
    log_file = open(log_file_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    print(f"日志文件保存路径: {log_file_path}")
    print("="*60)
    
    try:
        # 示例运行：使用仓库内示例拓扑（时间驱动版本）
        # 使用相对于脚本目录的路径
        sn_path = os.path.join(script_dir, 'topo', 'SN_topology.json')
        workflow_types = {
            'workflow1': os.path.join(script_dir, 'Workflow_topo', 'workflow1_topo.json'),
            # 可扩展：'workflow2': os.path.join(script_dir, 'Workflow_topo', 'workflow2_topo.json'), ...
        }
        USE_FINETUNING_MODEL = False
        
        # 获取模型路径
        policy_ckpt_path, value_ckpt_path = get_model_paths(script_dir, USE_FINETUNING_MODEL)
        
        training_stats, agent = run_ppo_batch_training(
            sn_topology_path=sn_path,
            workflow_types=workflow_types,
            #policy_ckpt=None,  # 使用随机初始化
            policy_ckpt=policy_ckpt_path,  # 如果文件存在则使用预训练模型
            value_ckpt=value_ckpt_path,  # 如果文件存在则使用预训练价值网络
            device='cpu',
            arrival_rate=0.1,   # arrival_rate = 0.2 表示每5个时间单位到达1个任务
            mean_lifetime=50.0,
            max_arrived_tasks=10,
            max_time_steps=2000,
            num_episodes_per_update=2,  # 批量大小：收集多少个episode的轨迹数据后再进行一次PPO更新
            train_iters=5,  # PPO更新迭代次数
            num_updates=30,  # 批量更新次数：总共执行多少次批量更新（即训练轮数）
        
            verbose = False  
        )
        
        print("\n批量训练统计:")
        for stat in training_stats:
            print(f"  更新 {stat['update_idx']+1}: 平均每个 episode 的最终回报={stat['avg_return']:.2f}, "
                  f"接受率={stat['avg_accepted']/stat['avg_arrived']:.2%}, "
                  f"样本数={stat['total_samples']}")
        
        # 保存训练结果、模型参数和可视化图表
        save_training_results(
            training_stats=training_stats,
            policy=agent.policy,
            value_net=agent.value_net,
            output_dir=None,  # 使用默认的相对路径
            run_dir=run_dir  # 使用已创建的运行目录
        )
        
        print(f"\n所有日志已保存到: {log_file_path}")
        
    finally:
        # 恢复stdout并关闭日志文件
        sys.stdout = original_stdout
        log_file.close()
        print(f"日志文件已关闭: {log_file_path}")


