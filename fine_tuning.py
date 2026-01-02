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

from model_1 import SimuVNE
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


class PPOAgent:

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
    episode_seed: int = None,
    verbose: bool = False):
    """
    运行一个PPO episode
    
    Args:
        agent: PPO智能体
        sn_topology_path: SN拓扑文件路径
        workflow_types: workflow类型字典
        device: 设备
        arrival_rate: 泊松到达率
        mean_lifetime: 平均生存时间
        max_arrived_tasks: 最大到达任务数
        max_time_steps: 最大时间步数
        update_after_episode: 是否在episode结束后立即更新
        episode_seed: episode随机种子
    
    Returns:
        episode统计数据
    """

def run_ppo_batch_training(
    sn_topology_path: str,
    workflow_types: Dict[str, str],
    policy_ckpt: str = None,
    value_ckpt: str = None,
    device: str = 'cpu',
    arrival_rate: float = 0.05,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 1000,
    num_episodes_per_update: int = 4,
    train_iters: int = 5,
    num_updates: int = 10,
    verbose: bool = False):
    """
    批量PPO训练

    """
 
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
    print(f"✓ 训练曲线图（总回报）已保存: {plot_path1}")
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
    print(f"✓ 训练曲线图（接受率）已保存: {plot_path2}")
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
    print(f"✓ 训练曲线图（平均r_t）已保存: {plot_path3}")
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
    print(f"✓ 策略网络（最新）已保存: {latest_policy_path}")
    
    latest_value_path = os.path.join(output_dir, 'value_network_latest.pth')
    torch.save({
        'model_state_dict': value_net.state_dict(),
    }, latest_value_path)
    print(f"✓ 价值网络（最新）已保存: {latest_value_path}")
    
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
            print(f"  ✓ 使用微调后的最新模型: {policy_ckpt_path}")
            return policy_ckpt_path, value_ckpt_path
        else:
            print(f"  ⚠️  微调模型不存在，使用随机初始化")
            return None, None
    else:
        # 使用 pretrain_outputs 目录下的预训练模型
        pretrain_policy_path = os.path.join(script_dir, 'pretrain_outputs', 'checkpoint_latest.pt')
        if os.path.exists(pretrain_policy_path):
            policy_ckpt_path = pretrain_policy_path
            value_ckpt_path = None  # pretrain_outputs目录下没有单独的价值网络
            print(f"  ✓ 使用预训练模型: {policy_ckpt_path}")
            return policy_ckpt_path, value_ckpt_path
        else:
            print(f"  ⚠️  预训练模型不存在，使用随机初始化")
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
        sn_path = os.path.join(script_dir, 'topo', 'SN_topology_2.json')
        workflow_types = {
            'workflow1': os.path.join(script_dir, 'workflow_topo', 'workflow1_topo.json'),
            # 可扩展：'workflow2': os.path.join(script_dir, 'workflow_topo', 'workflow2_topo.json'), ...
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
            mean_lifetime=1000.0,
            max_arrived_tasks=7,
            max_time_steps=3000,
            num_episodes_per_update=1,  # 批量大小：收集多少个episode的轨迹数据后再进行一次PPO更新
            train_iters=3,  # PPO更新迭代次数
            num_updates=44,  # 批量更新次数：总共执行多少次批量更新（即训练轮数）
        
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


