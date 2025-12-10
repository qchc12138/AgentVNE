"""
    最新的微调脚本
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

    def _generate_priority_lists(self, probs_matrix: torch.Tensor) -> List[List[int]]:
        """从概率矩阵采样生成优先级列表（逐步排除方式）"""
        N_v, N_s = probs_matrix.shape
        priority_lists = []
        
        for i in range(N_v):
            probs = probs_matrix[i].clone()  # [N_s] 复制原始概率
            priority_list = []
            remaining_indices = list(range(N_s))  # 剩余可选的SN节点索引
            
            # 逐步采样：每次采样一个节点，排除它，然后对剩余节点重新softmax
            while len(remaining_indices) > 0:
                # 对剩余节点的概率进行softmax归一化
                remaining_probs = probs[remaining_indices]
                remaining_probs_normalized = F.softmax(remaining_probs, dim=0)
                
                # 从归一化后的概率分布中采样
                cat = Categorical(probs=remaining_probs_normalized)
                sampled_idx_in_remaining = cat.sample().item()
                
                # 获取实际的SN节点索引
                actual_sn_idx = remaining_indices[sampled_idx_in_remaining]
                priority_list.append(actual_sn_idx)
                
                # 从剩余列表中移除已选中的节点
                remaining_indices.remove(actual_sn_idx)
            
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
    def _act_original(self, vn: Data, sn: Data) -> Tuple[Dict[int, int], torch.Tensor, torch.Tensor, Data]:
        """原始的随机采样策略（向后兼容）"""
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
        return mapping, torch.tensor(logprob_sum, device=self.device, dtype=torch.float), value, sn

    @torch.no_grad()
    def act(self, vn: Data, sn: Data, env: Optional[SimuVNEEnv] = None, k_hop: int = 1, verbose: bool = False) -> Tuple[Dict[int, int], torch.Tensor, torch.Tensor, Data]:
        """
        放置策略：基于模型概率采样优先级 + BFS/k-hop 资源检查（无 NodeRank）。
        """
        # 如果没有提供env，使用原来的随机采样策略（向后兼容）
        if env is None:
            return self._act_original(vn, sn)

        vn = vn.to(self.device)
        sn = sn.to(self.device)
        
        # 在策略网络前向传播前，临时修改SN的CPU特征为 (cpu_res + bias_cpu)
        if env is not None:
            sn_with_bias = self._apply_bias_to_sn_features(sn, env)
        else:
            sn_with_bias = sn
        
        probs_matrix = self.policy(vn, sn_with_bias)  # [N_v, N_s]
        N_v, N_s = probs_matrix.shape

        # 打印模型输出的概率矩阵
        print(f"\n【Workflow到达】模型输出的概率矩阵 [VN节点数={N_v}, SN节点数={N_s}]:")
        print(f"概率矩阵:\n{probs_matrix.detach().cpu().numpy()}")

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
                    print(f"  [放置] VN节点{vn_idx} → SN节点{constraint_node_id} (约束节点)")
                    print(f"    SN节点{constraint_node_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                else:
                    self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
                    value = self.value_net(vn, sn_with_bias)
                    return {}, torch.tensor(0.0, device=self.device, dtype=torch.float), value, sn_with_bias

        if len(constraint_placed_vn) == N_v:
            value = self.value_net(vn, sn_with_bias)
            return mapping, torch.tensor(0.0, device=self.device, dtype=torch.float), value, sn_with_bias

        # 生成优先级列表（模型概率采样，无 NodeRank）
        priority_lists = self._generate_priority_lists(probs_matrix)
        
        # 打印采样得到的优先级列表
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
            value = self.value_net(vn, sn)
            return mapping, torch.tensor(0.0, device=self.device, dtype=torch.float), value
        first_vn = max(non_constraint_vn, key=lambda i: vn_resource_demands[i])

        placed_first = False
        for first_sn_idx in priority_lists[first_vn]:
            first_sn_id = sn_node_list[first_sn_idx]
            if self._check_and_deduct_resource(env, first_sn_id, first_vn, vn, verbose=False):
                mapping[first_vn] = first_sn_id
                resource_deduction_history.append((first_sn_id, first_vn))
                placed_first = True
                sn_node = env.G_sn.nodes[first_sn_id]
                print(f"  [放置] VN节点{first_vn} → SN节点{first_sn_id} (首个非约束节点)")
                print(f"    SN节点{first_sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                break
        if not placed_first:
            self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
            value = self.value_net(vn, sn)
            return {}, torch.tensor(0.0, device=self.device, dtype=torch.float), value

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
                                print(f"  [放置] VN节点{u} → SN节点{sn_id} (k={k}跳邻居)")
                                print(f"    SN节点{sn_id}资源容量: CPU={sn_node['cpu_res']:.3f}, MEM={sn_node['mem_res']:.3f}, DISK={sn_node['disk_res']:.3f}")
                                break
                        k += 1
                    if not placed:
                        self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
                        value = self.value_net(vn, sn_with_bias)
                        return {}, torch.tensor(0.0, device=self.device, dtype=torch.float), value, sn_with_bias

            queue = sorted(
                [i for i in new_placed if i not in constraint_placed_vn],
                key=lambda i: (vn_degrees[i], vn_resource_demands[i]),
                reverse=True
            )

        if len(placed_vn) < N_v:
            self._rollback_resource_deductions(env, resource_deduction_history, vn, verbose=False)
            value = self.value_net(vn, sn_with_bias)
            return {}, torch.tensor(0.0, device=self.device, dtype=torch.float), value, sn_with_bias

        # 计算 logprob 和 value
        logprob_sum = torch.tensor(0.0, device=self.device, dtype=torch.float)
        sn_id_to_idx = {sn_id: idx for idx, sn_id in enumerate(sn_node_list)}
        for vn_idx, sn_id in mapping.items():
            probs = probs_matrix[vn_idx]
            cat = Categorical(probs=probs)
            sn_idx = sn_id_to_idx.get(sn_id, sn_id % N_s)
            logprob_sum += cat.log_prob(torch.tensor(sn_idx, device=self.device))

        value = self.value_net(vn, sn_with_bias)
        return mapping, logprob_sum, value, sn_with_bias

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 按时间展开一次episode（支持批量更新时多个episode拼接）
        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float, device=self.device)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 0.0 if dones[t] else 1.0
            # 修复：如果episode结束（dones[t]=True）或到达最后一个时间步，next_value=0
            # 这样可以正确处理批量更新时多个episode拼接的情况
            if t == T - 1 or dones[t]:
                next_value = 0.0
            else:
                next_value = float(values[t + 1])
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
        if not isinstance(values_list, list):
            values_list = [values_list]
        elif len(values_list) > 0 and isinstance(values_list[0], list):
            values_list = [v[0] if isinstance(v, list) else v for v in values_list]
        
        adv, rets = self.compute_gae(rewards.tolist(), values_list, dones)

        for iter_idx in range(train_iters):
            print(f"      [PPO迭代 {iter_idx+1}/{train_iters}]", end=' ')
            new_logprobs: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []
            for vn, sn, mapping in zip(vn_list, sn_list, mappings):
                vn = vn.to(self.device)
                sn = sn.to(self.device)
                probs_matrix = self.policy(vn, sn)
                N_s = probs_matrix.shape[1]

                if len(mapping) > 0:
                    all_sn_ids = sorted(set(mapping.values()))
                    min_sn_id = min(all_sn_ids)
                    max_sn_id = max(all_sn_ids)
                    if min_sn_id >= 0 and max_sn_id < N_s:
                        sn_id_to_idx = {sn_id: sn_id for sn_id in all_sn_ids}
                    elif max_sn_id - min_sn_id < N_s:
                        sn_id_to_idx = {sn_id: sn_id - min_sn_id for sn_id in all_sn_ids}
                    else:
                        sn_id_to_idx = {sn_id: sn_id % N_s for sn_id in all_sn_ids}
                else:
                    sn_id_to_idx = {}

                lp_sum = torch.tensor(0.0, device=self.device)
                ent_sum = torch.tensor(0.0, device=self.device)
                for vn_idx, sn_id in mapping.items():
                    sn_idx = sn_id_to_idx.get(sn_id, sn_id % N_s)
                    if 0 <= sn_idx < probs_matrix.shape[1]:
                        cat = Categorical(probs=probs_matrix[vn_idx])
                        lp_sum = lp_sum + cat.log_prob(torch.tensor(sn_idx, device=self.device))
                        ent_sum = ent_sum + cat.entropy()

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

            v_preds = []
            for vn, sn in zip(vn_list, sn_list):
                v_preds.append(self.value_net(vn.to(self.device), sn.to(self.device)))
            v_preds = torch.stack(v_preds)
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


# 已废弃：_rebuild_sn_state_with_bias 函数
# 现在bias的应用在PPOAgent._apply_bias_to_sn_features()中完成，只在策略网络前向传播时临时使用
# 这样可以避免修改sn_state对象，更清晰地表达"只在神经网络输入时临时使用bias"的意图


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
            
            # 获取当前SN状态（包含剩余资源）
            sn_state = env.get_sn_state()
            
            # 调用策略网络生成放置方案（一次性采样，传入env以使用新的放置策略）
            # 注意：bias会在agent.act()内部临时应用到SN特征上，不会修改sn_state
            # act()内部会打印概率矩阵、优先级列表和每次放置的映射
            mapping, logprob, value, sn_with_bias = agent.act(vn, sn_state, env=env, k_hop=1, verbose=False)
            
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
            
            # 打印放置结果
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
                'done': False,
            })
            
            traj_vn.append(vn)
            traj_sn.append(sn_with_bias)  # 保存 sn_with_bias 而不是原始 sn_state
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
    
    # Episode完成（不打印，精简输出）
    
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
        # 转换为绝对路径
        if not os.path.isabs(policy_ckpt):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            policy_ckpt = os.path.join(script_dir, policy_ckpt)
        
        # 检查文件是否存在
        if not os.path.exists(policy_ckpt):
            print(f"  ⚠️  警告: 预训练模型文件不存在: {policy_ckpt}")
            print(f"  使用随机初始化的策略网络")
        else:
            try:
                ckpt = torch.load(policy_ckpt, map_location='cpu', weights_only=False)
                state_dict = ckpt.get('model_state_dict', ckpt)
                policy.load_state_dict(state_dict, strict=False)
                print(f"  ✓ 加载预训练模型: {policy_ckpt}")
            except Exception as e:
                print(f"  ⚠️  警告: 加载预训练模型失败: {e}")
                print(f"  使用随机初始化的策略网络")
    else:
        print(f"  ✓ 使用随机初始化的策略网络")
    value_net = ValueNet()
    agent = PPOAgent(policy, value_net, device=device)
    print(f"  ✓ PPO Agent创建完成 (设备: {device})")
    
    training_stats = []
    
    for update_idx in range(num_updates):
        # 批量更新（不打印标题，精简输出）
        
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
            
            # Episode完成（不打印，精简输出）
        
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
            
            print(f"  【更新 {update_idx + 1}/{num_updates} 结果】平均每个 episode 的最终回报: {avg_return:.2f}, 平均接受率: {avg_accepted/avg_arrived:.2%} ({avg_accepted:.1f}/{avg_arrived:.1f})")
            
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
                          output_dir: str = None):
    """
    保存训练结果、模型参数和可视化图表
    
    Args:
        training_stats: 训练统计信息列表
        policy: 策略网络
        value_net: 价值网络
        output_dir: 输出目录（如果为None，使用相对于脚本目录的默认路径）
    """
    # 如果没有指定输出目录，使用默认的相对路径
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'finetuning_putput')
    
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
    
    # 2. 绘制训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO Training Results', fontsize=16, fontweight='bold')
    
    # 子图1: 平均每个 episode 的最终回报变化
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
    # 获取脚本所在目录，用于构建相对路径的默认值
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 示例运行：使用仓库内示例拓扑（时间驱动版本）
    # 使用相对于脚本目录的路径
    sn_path = os.path.join(script_dir, 'topo', 'SN_topology_2.json')
    workflow_types = {
        'workflow1': os.path.join(script_dir, 'workflow_topo', 'workflow1_topo.json'),
        # 可扩展：'workflow2': os.path.join(script_dir, 'workflow_topo', 'workflow2_topo.json'), ...
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
    # 不打印标题，精简输出
    
    # 预训练模型路径（可选，如果文件不存在则使用随机初始化）
    policy_ckpt_path = os.path.join(script_dir, 'pretrain_outputs', 'checkpoint_latest.pt')
    
    training_stats, agent = run_ppo_batch_training(
        sn_topology_path=sn_path,
        workflow_types=workflow_types,
        #policy_ckpt=None,  # 使用随机初始化
        policy_ckpt=policy_ckpt_path if os.path.exists(policy_ckpt_path) else None,  # 如果文件存在则使用预训练模型
        device='cpu',
        arrival_rate=1,   # arrival_rate = 0.2 表示每5个时间单位到达1个任务
        mean_lifetime=10.0,
        max_arrived_tasks=15,
        max_time_steps=2000,
        num_episodes_per_update=1,  # 批量大小：收集多少个episode的轨迹数据后再进行一次PPO更新
                                     # 例如：1表示每个episode结束后立即更新；4表示收集4个episode后统一更新
        train_iters=3,  # PPO更新迭代次数：每次批量更新时，对策略网络和价值网络进行多少次梯度更新
                         # 例如：3表示每次批量更新时，策略和价值网络各更新3次
        num_updates=20  # 批量更新次数：总共执行多少次批量更新（即训练轮数）
                        # 例如：1表示只执行1次批量更新；30表示执行30次批量更新
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
        output_dir=None  # 使用默认的相对路径
    )


