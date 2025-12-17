"""
    测试脚本：使用训练好的模型进行测试
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
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

    def _generate_priority_lists(self, probs_matrix: torch.Tensor, seed: Optional[int] = None) -> List[List[int]]:
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

    def _get_vn_neighbors(self, vn: Data) -> Dict[int, Set[int]]:
        """获取VN节点的邻居关系（无向，即使边是有向的）"""
        neighbors = {i: set() for i in range(vn.x.size(0))}
        edge_index = vn.edge_index
        for i in range(edge_index.size(1)):
            u = int(edge_index[0, i].item())
            v = int(edge_index[1, i].item())
            neighbors[u].add(v)  # u的邻居包括v
            neighbors[v].add(u)  # v的邻居包括u（即使边是有向的）
        return neighbors

    def _get_sn_k_hop_neighbors(self, env: SimuVNEEnv, sn_node_id: int, k: int) -> Set[int]:
        """获取SN节点的k跳邻居（包括k跳内的所有节点）"""
        if k == 0:
            return {sn_node_id}
        # 使用networkx的single_source_shortest_path_length
        paths = nx.single_source_shortest_path_length(env.G_sn, sn_node_id, cutoff=k)
        return set(paths.keys())  # 包括距离0到k的所有节点

    def _apply_bias_to_sn_features(self, sn: Data, env: SimuVNEEnv) -> Data:
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

    def _check_and_deduct_resource(self, env: SimuVNEEnv, sn_node_id: int, vn_node_idx: int, vn: Data, verbose: bool = False) -> bool:
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
    
    def _rollback_resource_deductions(self, env: SimuVNEEnv, deduction_history: List[Tuple[int, int]], vn: Data, verbose: bool = False):
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

    def _check_sn_resource(self, env: SimuVNEEnv, sn_node_id: int, vn_node_idx: int, vn: Data, 
                          temp_mapping: Optional[Dict[int, int]] = None, verbose: bool = False) -> bool:
        """
        检查SN节点是否有足够资源放置VN节点
        
        Args:
            env: 环境对象
            sn_node_id: SN节点ID
            vn_node_idx: VN节点索引
            vn: VN图数据
            temp_mapping: 临时映射（当前轮已放置的节点），用于虚拟扣减资源
            verbose: 是否打印详细信息
        """
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
        
        if verbose:
            print(f"        [资源检查] VN节点{vn_node_idx} → SN节点{sn_node_id}")
            print(f"          需求: CPU={cpu_demand:.3f}, MEM={mem_demand:.3f}, DISK={disk_demand:.3f}")
            print(f"          初始可用: CPU={available_cpu:.3f}, MEM={available_mem:.3f}, DISK={available_disk:.3f}")
        
        if temp_mapping:
            # 虚拟扣减当前轮已放置在该SN节点上的VN节点的资源
            temp_deduct_cpu = 0.0
            temp_deduct_mem = 0.0
            temp_deduct_disk = 0.0
            for vn_idx, sn_id in temp_mapping.items():
                if sn_id == sn_node_id:
                    vn_feats_temp = vn.x[vn_idx]
                    cpu_temp = float(vn_feats_temp[0].item()) * (env._sn_max_capacity['cpu_max'] + 1e-8)
                    mem_temp = float(vn_feats_temp[1].item()) * (env._sn_max_capacity['mem_max'] + 1e-8)
                    disk_temp = float(vn_feats_temp[2].item()) * (env._sn_max_capacity['disk_max'] + 1e-8)
                    available_cpu -= cpu_temp
                    available_mem -= mem_temp
                    available_disk -= disk_temp
                    temp_deduct_cpu += cpu_temp
                    temp_deduct_mem += mem_temp
                    temp_deduct_disk += disk_temp
            
            if verbose and (temp_deduct_cpu > 0 or temp_deduct_mem > 0 or temp_deduct_disk > 0):
                print(f"          临时扣减 (temp_mapping={list(temp_mapping.keys())}): CPU={temp_deduct_cpu:.3f}, MEM={temp_deduct_mem:.3f}, DISK={temp_deduct_disk:.3f}")
        
        if verbose:
            print(f"          最终可用: CPU={available_cpu:.3f}, MEM={available_mem:.3f}, DISK={available_disk:.3f}")
        
        # 检查剩余资源
        cpu_ok = cpu_demand <= available_cpu + 1e-9
        mem_ok = mem_demand <= available_mem + 1e-9
        disk_ok = disk_demand <= available_disk + 1e-9
        
        if verbose:
            result = "✓通过" if (cpu_ok and mem_ok and disk_ok) else "✗失败"
            print(f"          结果: {result} (CPU:{'✓' if cpu_ok else '✗'}, MEM:{'✓' if mem_ok else '✗'}, DISK:{'✓' if disk_ok else '✗'})")
        
        if not cpu_ok:
            return False
        if not mem_ok:
            return False
        if not disk_ok:
            return False
        return True


    @torch.no_grad()
    def act(self, vn: Data, sn: Data, env: Optional[SimuVNEEnv] = None, k_hop: int = 1, verbose: bool = False, seed: Optional[int] = None) -> Tuple[Dict[int, int], Data]:
        """
        放置策略：基于模型概率采样优先级 + BFS/k-hop 资源检查（无 NodeRank）。
        
        Args:
            vn: VN图数据
            sn: SN图数据
            env: 环境对象
            k_hop: k跳搜索参数
            verbose: 是否打印详细信息
            seed: 随机数种子（用于采样，None则使用默认随机数）
        """
        # 测试模式必须提供env
        if env is None:
            raise ValueError("测试模式必须提供env参数")

        vn = vn.to(self.device)
        sn = sn.to(self.device)
        
        # 在策略网络前向传播前，临时修改SN的CPU特征为 (cpu_res + bias_cpu)
        if env is not None:
            sn_with_bias = self._apply_bias_to_sn_features(sn, env)
        else:
            sn_with_bias = sn
        
        probs_matrix = self.policy(vn, sn_with_bias)  # [N_v, N_s]
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
                if self._check_and_deduct_resource(env, constraint_node_id, vn_idx, vn, verbose=False):
                    mapping[vn_idx] = constraint_node_id
                    constraint_placed_vn.add(vn_idx)
                    resource_deduction_history.append((constraint_node_id, vn_idx))
                    sn_node = env.G_sn.nodes[constraint_node_id]
                    if verbose:
                        print(f"  [放置] VN节点{vn_idx} → SN节点{constraint_node_id} (约束节点)")
                        print(f"    SN节点{constraint_node_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                else:
                    self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
                    return {}, sn_with_bias

        if len(constraint_placed_vn) == N_v:
            return mapping, sn_with_bias

        # 生成优先级列表（按概率降序排序，测试模式）
        priority_lists = self._generate_priority_lists(probs_matrix)
        
        # 打印采样得到的优先级列表（如果verbose=True）
        if verbose:
            print(f"\n【采样得到的优先级列表】:")
            for vn_idx in range(N_v):
                priority_sn_ids = [sn_node_list[idx] for idx in priority_lists[vn_idx]]
                priority_probs = [float(probs_matrix[vn_idx][idx].item()) for idx in priority_lists[vn_idx]]
                print(f"  VN节点{vn_idx}: SN节点优先级序列 = {priority_sn_ids}")
                print(f"    对应概率值 = {[f'{p:.4f}' for p in priority_probs]}")

        # VN资源需求与度
        vn_neighbors = self._get_vn_neighbors(vn)
        vn_degrees = {i: len(vn_neighbors[i]) for i in range(N_v)}
        vn_resource_demands = {i: float(vn.x[i][0] + vn.x[i][1] + vn.x[i][2]) for i in range(N_v)}

        # 选择首个非约束 VN（资源需求最大）
        non_constraint_vn = [i for i in range(N_v) if i not in constraint_placed_vn]
        if not non_constraint_vn:
            return mapping, sn_with_bias
        first_vn = max(non_constraint_vn, key=lambda i: vn_resource_demands[i])

        placed_first = False
        for first_sn_idx in priority_lists[first_vn]:
            first_sn_id = sn_node_list[first_sn_idx]
            if self._check_and_deduct_resource(env, first_sn_id, first_vn, vn, verbose=False):
                mapping[first_vn] = first_sn_id
                resource_deduction_history.append((first_sn_id, first_vn))
                placed_first = True
                sn_node = env.G_sn.nodes[first_sn_id]
                if verbose:
                    print(f"  [放置] VN节点{first_vn} → SN节点{first_sn_id} (首个非约束节点)")
                    print(f"    SN节点{first_sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                break
        if not placed_first:
            self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
            return {}, sn_with_bias

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
                    if self._check_and_deduct_resource(env, vi_sn_id, u, vn, verbose=False):
                        mapping[u] = vi_sn_id
                        placed_vn.add(u)
                        new_placed.append(u)
                        resource_deduction_history.append((vi_sn_id, u))
                        sn_node = env.G_sn.nodes[vi_sn_id]
                        if verbose:
                            print(f"  [放置] VN节点{u} → SN节点{vi_sn_id} (同SN节点)")
                            print(f"    SN节点{vi_sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                        continue

                    # k-hop 搜索
                    k = 1
                    max_k = len(sn_node_list)
                    placed = False
                    while k <= max_k and not placed:
                        k_hop_neighbors = self._get_sn_k_hop_neighbors(env, vi_sn_id, k)
                        for sn_idx in priority_lists[u]:
                            sn_id = sn_node_list[sn_idx]
                            if sn_id in k_hop_neighbors and self._check_and_deduct_resource(env, sn_id, u, vn, verbose=False):
                                mapping[u] = sn_id
                                placed_vn.add(u)
                                new_placed.append(u)
                                resource_deduction_history.append((sn_id, u))
                                placed = True
                                sn_node = env.G_sn.nodes[sn_id]
                                if verbose:
                                    print(f"  [放置] VN节点{u} → SN节点{sn_id} (k={k}跳邻居)")
                                    print(f"    SN节点{sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                                break
                        k += 1
                    if not placed:
                        self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
                        return {}, sn_with_bias

            queue = sorted(
                [i for i in new_placed if i not in constraint_placed_vn],
                key=lambda i: (vn_degrees[i], vn_resource_demands[i]),
                reverse=True
            )

        if len(placed_vn) < N_v:
            self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
            return {}, sn_with_bias

        # 测试模式：不需要计算logprob和value
        return mapping, sn_with_bias



# 已废弃：_rebuild_sn_state_with_bias 函数
# 现在bias的应用在PPOAgent._apply_bias_to_sn_features()中完成，只在策略网络前向传播时临时使用
# 这样可以避免修改sn_state对象，更清晰地表达"只在神经网络输入时临时使用bias"的意图


def run_test_episode(
    agent: PPOAgent,
    sn_topology_path: str,
    workflow_types: Dict[str, str],
    device: str = 'cpu',
    arrival_rate: float = 0.05,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 1000,
    verbose: bool = False):
    """
    运行一个测试episode（时间驱动版本）：
    - 按时间单位推进
    - 泊松到达控制任务生成
    - 指数分布控制任务生存时间
    - 收集指定数量的任务到达后结束
    
    Args:
        agent: PPO智能体
        sn_topology_path: SN拓扑文件路径
        workflow_types: workflow类型字典
        device: 设备
        arrival_rate: 泊松到达率
        mean_lifetime: 平均生存时间
        max_arrived_tasks: 最大到达任务数
        max_time_steps: 最大时间步数
        verbose: 是否打印详细信息
    
    Returns:
        episode统计数据
    """

    # 构建环境与任务生成器
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
        seed=42,  # 固定种子，使所有episode的workflow轨迹相同（任务到达时间、类型、生存时间）
        sn_capacity_for_norm=sn_capacity
    )

    # 时间驱动主循环
    
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
            
            # 获取当前SN状态（包含剩余资源）
            sn_state = env.get_sn_state()
            
            # 调用策略网络生成放置方案（测试模式：按概率降序排序）
            # 注意：bias会在agent.act()内部临时应用到SN特征上，不会修改sn_state
            # act()内部会打印概率矩阵、优先级列表和每次放置的映射（如果verbose=True）
            mapping, sn_with_bias = agent.act(vn, sn_state, env=env, k_hop=1, verbose=verbose)
            
            # 检查是否成功放置（所有节点都已映射，资源已在act()中扣减）
            if len(mapping) == vn.x.size(0):
                # 所有节点都已放置，资源已在act()中扣减，只需要添加到存活集合
                vn_paths = env._compute_paths_and_bw_demand(vn, mapping)
                if vn_paths is None:
                    # 路径不存在，需要回滚资源
                    # 根据mapping构建回滚历史
                    rollback_history = [(sn_id, vn_idx) for vn_idx, sn_id in mapping.items()]
                    agent._rollback_resource_deductions(env, rollback_history, vn, verbose=False)
                    success, r_t = False, env.penalty
                else:
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
            else:
                # 部分节点未放置，资源已在act()中回滚，返回失败
                success, r_t = False, env.penalty
            
            # 打印放置结果（如果verbose=True）
            if verbose:
                status = "✓成功" if success else "✗失败"
                print(f"\n【放置结果】任务 #{task_id}: {status}, r_t={r_t:.3f}")
                print(f"【存活任务数】当前底层网络中存活的任务总数: {len(env.active_workflows)}")
                print("="*60)
            
            # 记录轨迹
            env.traj.append({
                'time': env.current_time,
                'task_id': task_id,
                'success': success,
                'r_t': r_t,
                'active_tasks': len(env.active_workflows),  # 记录当前存活任务数量
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
            
            # 为了保持轨迹连续，使用零填充（可选）
            # 这里不添加到训练轨迹，仅记录在env.traj中
        
        time_step += 1
    
    # 标记结束
    if len(env.traj) > 0:
        env.traj[-1]['done'] = True
    
    # 计算最终回报
    final_R = env.compute_final_return()
    
    # 计算每个episode的平均r_t（所有时间步的r_t的平均值）
    # 从env.traj中提取所有时间步的r_t（包括有任务到达和无任务到达的时间步）
    all_rt_values = [traj_entry['r_t'] for traj_entry in env.traj if traj_entry.get('r_t') is not None]
    avg_rt_per_episode = 0.0
    if len(all_rt_values) > 0:
        avg_rt_per_episode = sum(all_rt_values) / len(all_rt_values)
    
    # 提取每个任务到来时的r_t和存活任务数量（只包括有任务到达的时间步）
    task_rt_info = []
    for traj_entry in env.traj:
        if traj_entry.get('task_id') is not None and traj_entry.get('r_t') is not None:
            r_t = traj_entry['r_t']
            active_tasks = traj_entry.get('active_tasks', 0)
            task_rt_info.append((r_t, active_tasks))
    
    # 打印每个任务到来时的r_t列表（格式：r_t : 当前存活任务数量）
    if len(task_rt_info) > 0:
        print(f"\n【Episode结束】本episode共处理 {len(task_rt_info)} 个任务")
        print(f"每个任务到来时的r_t列表（格式：r_t : 当前存活任务数量）:")
        # 将所有任务的信息打印成一横排
        rt_str_list = [f"{r_t:.3f} : {active_tasks}" for r_t, active_tasks in task_rt_info]
        print("  " + " ，  ".join(rt_str_list))
        avg_rt = sum(rt for rt, _ in task_rt_info) / len(task_rt_info)
        print(f"r_t均值: {avg_rt:.3f}")
    
    result = {
        'final_return': final_R,
        'avg_rt': avg_rt_per_episode,  # 每个episode的平均r_t
        'arrived': env.arrived_count,
        'accepted': env.accepted_count,
        'time_steps': time_step,
    }
    
    return result


def run_test(
    sn_topology_path: str,
    workflow_types: Dict[str, str],
    policy_ckpt: str = None,
    device: str = 'cpu',
    arrival_rate: float = 0.05,
    mean_lifetime: float = 10.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 1000,
    num_episodes: int = 1,
    verbose: bool = False):
    """
    运行测试：使用训练好的模型进行测试
    
    Args:
        sn_topology_path: SN拓扑文件路径
        workflow_types: workflow类型字典
        policy_ckpt: 策略网络路径（可选）
        device: 设备
        arrival_rate: 泊松到达率
        mean_lifetime: 平均生存时间
        max_arrived_tasks: 每个episode最大到达任务数
        max_time_steps: 每个episode最大时间步数
        num_episodes: 运行多少个episode
        verbose: 是否打印详细信息
    
    Returns:
        test_stats: 测试统计信息列表
        agent: PPOAgent对象
    """
    # 初始化策略网络
    print(f"\n【初始化】创建策略网络...")
    policy = SimuVNE()
    if policy_ckpt:
        # 转换为绝对路径
        if not os.path.isabs(policy_ckpt):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            policy_ckpt = os.path.join(script_dir, policy_ckpt)
        
        # 检查文件是否存在
        if not os.path.exists(policy_ckpt):
            print(f"  ⚠️  警告: 模型文件不存在: {policy_ckpt}")
            print(f"  使用随机初始化的策略网络")
        else:
            try:
                ckpt = torch.load(policy_ckpt, map_location='cpu', weights_only=False)
                state_dict = ckpt.get('model_state_dict', ckpt)
                policy.load_state_dict(state_dict, strict=False)
                print(f"  ✓ 加载模型: {policy_ckpt}")
            except Exception as e:
                print(f"  ⚠️  警告: 加载模型失败: {e}")
                print(f"  使用随机初始化的策略网络")
    else:
        print(f"  ⚠️  警告: 未指定模型路径，使用随机初始化的策略网络")
    
    # 测试模式不需要价值网络，但为了兼容PPOAgent，创建一个
    value_net = ValueNet()
    agent = PPOAgent(policy, value_net, device=device)
    print(f"  ✓ Agent创建完成 (设备: {device})")
    
    test_stats = []
    
    for ep_idx in range(num_episodes):
        print(f"\n【测试Episode {ep_idx + 1}/{num_episodes}】")
        result = run_test_episode(
            agent=agent,
            sn_topology_path=sn_topology_path,
            workflow_types=workflow_types,
            device=device,
            arrival_rate=arrival_rate,
            mean_lifetime=mean_lifetime,
            max_arrived_tasks=max_arrived_tasks,
            max_time_steps=max_time_steps,
            verbose=verbose
        )
        
        test_stats.append({
            'episode_idx': ep_idx,
            'final_return': result['final_return'],
            'avg_rt': result['avg_rt'],
            'arrived': result['arrived'],
            'accepted': result['accepted'],
            'acceptance_rate': result['accepted'] / result['arrived'] if result['arrived'] > 0 else 0.0,
        })
        
        print(f"  Episode {ep_idx + 1} 结果: 最终回报={result['final_return']:.2f}, "
              f"接受率={result['accepted']/result['arrived']:.2%} ({result['accepted']}/{result['arrived']}), "
              f"平均r_t={result['avg_rt']:.3f}")
    
    # 计算总体统计
    if len(test_stats) > 0:
        avg_return = sum(s['final_return'] for s in test_stats) / len(test_stats)
        avg_acceptance_rate = sum(s['acceptance_rate'] for s in test_stats) / len(test_stats)
        avg_rt = sum(s['avg_rt'] for s in test_stats) / len(test_stats)
        print(f"\n【测试总结】")
        print(f"  总episode数: {len(test_stats)}")
        print(f"  平均最终回报: {avg_return:.2f}")
        print(f"  平均接受率: {avg_acceptance_rate:.2%}")
        print(f"  平均r_t: {avg_rt:.3f}")
    
    return test_stats, agent


def save_test_results(test_stats: List[Dict], 
                      output_dir: str = None,
                      run_dir: str = None):
    """
    保存测试结果
    
    Args:
        test_stats: 测试统计信息列表
        output_dir: 输出基础目录（如果为None，使用相对于脚本目录的默认路径）
        run_dir: 运行目录（如果指定，直接使用该目录；否则在output_dir下创建新的run_xxxxxx目录）
    """
    # 如果指定了run_dir，直接使用（推荐方式：每次测试创建独立的带时间戳文件夹）
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        # 从run_dir提取output_dir（用于保存latest模型）
        output_dir = os.path.dirname(run_dir)
        # 从run_dir提取timestamp（用于保存统计信息）
        timestamp = os.path.basename(run_dir).replace('run_', '')
    else:
        # 如果没有指定run_dir，自动创建带时间戳的文件夹
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'finetuning_test_output')
        
        # 转换为绝对路径
        if not os.path.isabs(output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, output_dir)
        
        # 创建输出目录（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"创建测试输出文件夹: {run_dir}")
    
    print(f"\n{'='*60}")
    print(f"保存测试结果到: {run_dir}")
    print(f"{'='*60}")
    
    # 保存测试统计数据（JSON格式）
    stats_path = os.path.join(run_dir, 'test_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'num_episodes': len(test_stats),
            'test_stats': test_stats,
            'summary': {
                'avg_return': sum(s['final_return'] for s in test_stats) / len(test_stats) if test_stats else 0.0,
                'avg_acceptance_rate': sum(s['acceptance_rate'] for s in test_stats) / len(test_stats) if test_stats else 0.0,
                'avg_rt': sum(s['avg_rt'] for s in test_stats) / len(test_stats) if test_stats else 0.0,
            }
        }, f, indent=2)
    print(f"✓ 测试统计已保存: {stats_path}")
    
    # 保存文本格式的测试摘要
    summary_path = os.path.join(run_dir, 'test_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Test Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Time: {timestamp}\n")
        f.write(f"Total Episodes: {len(test_stats)}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Test Results:\n")
        f.write("-"*60 + "\n")
        for s in test_stats:
            f.write(f"Episode {s['episode_idx']+1}: "
                   f"Return={s['final_return']:.2f}, "
                   f"Acceptance={s['acceptance_rate']:.2%}, "
                   f"Avg_r_t={s['avg_rt']:.3f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Summary:\n")
        f.write("-"*60 + "\n")
        if test_stats:
            avg_return = sum(s['final_return'] for s in test_stats) / len(test_stats)
            avg_acceptance_rate = sum(s['acceptance_rate'] for s in test_stats) / len(test_stats)
            avg_rt = sum(s['avg_rt'] for s in test_stats) / len(test_stats)
            f.write(f"Average Return: {avg_return:.2f}\n")
            f.write(f"Average Acceptance Rate: {avg_acceptance_rate:.2%}\n")
            f.write(f"Average r_t: {avg_rt:.3f}\n")
    
    print(f"✓ 测试摘要已保存: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"所有结果已成功保存到: {run_dir}")
    print(f"{'='*60}\n")
    
    return run_dir


def get_model_path(script_dir: str, use_finetuning_model: bool = True) -> Optional[str]:
    """
    根据 use_finetuning_model 参数获取模型路径
    
    Args:
        script_dir: 脚本所在目录
        use_finetuning_model: 设置为 True 使用微调后的最新模型（默认），False 使用预训练模型
        
    Returns:
        policy_ckpt_path: 策略网络路径
    """
    if use_finetuning_model:
        # 使用 finetuning_putput 目录下的最新模型（微调后的模型）
        finetuning_policy_path = os.path.join(script_dir, 'finetuning_putput', 'policy_network_latest.pth')
        if os.path.exists(finetuning_policy_path):
            print(f"  ✓ 使用微调后的最新模型: {finetuning_policy_path}")
            return finetuning_policy_path
        else:
            print(f"  ⚠️  微调模型不存在: {finetuning_policy_path}")
            return None
    else:
        # 使用 pretrain_outputs 目录下的预训练模型
        pretrain_policy_path = os.path.join(script_dir, 'pretrain_outputs', 'checkpoint_latest.pt')
        if os.path.exists(pretrain_policy_path):
            print(f"  ✓ 使用预训练模型: {pretrain_policy_path}")
            return pretrain_policy_path
        else:
            print(f"  ⚠️  预训练模型不存在: {pretrain_policy_path}")
            return None


if __name__ == '__main__':
    # 获取脚本所在目录，用于构建相对路径的默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建带时间戳的运行目录和日志文件（保存所有打印输出）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(script_dir, 'finetuning_test_output')
    os.makedirs(output_base_dir, exist_ok=True)
    run_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"创建测试输出文件夹: {run_dir}")
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
    
        # ========== 模型路径选择（通过参数控制）==========
        # 设置为 True 使用微调后的最新模型（默认），False 使用预训练模型
        USE_FINETUNING_MODEL = True
        
        # 获取模型路径
        policy_ckpt_path = get_model_path(script_dir, USE_FINETUNING_MODEL)
        
        # 运行测试
        test_stats, agent = run_test(
            sn_topology_path=sn_path,
            workflow_types=workflow_types,
            policy_ckpt=policy_ckpt_path,  # 如果文件存在则使用指定模型
            device='cpu',
            arrival_rate=0.2,   # arrival_rate = 0.2 表示每5个时间单位到达1个任务
            mean_lifetime=35.0,
            max_arrived_tasks=20,
            max_time_steps=3000,
            num_episodes=1,  # 运行多少个episode进行测试
            verbose=True  # 设置为True以打印详细信息
        )
        
        # 保存测试结果
        save_test_results(
            test_stats=test_stats,
            output_dir=None,  # 使用默认的相对路径
            run_dir=run_dir  # 使用已创建的运行目录
        )
        
        print(f"\n所有日志已保存到: {log_file_path}")
        
    finally:
        # 恢复stdout并关闭日志文件
        sys.stdout = original_stdout
        log_file.close()
        print(f"日志文件已关闭: {log_file_path}")


