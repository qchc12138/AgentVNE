"""
遗传算法核心实现：用于解决 VNE（Virtual Network Embedding）问题。

主要组件：
- Individual: 个体（染色体），表示一个 VN 到 SN 的映射方案
- GeneticAlgorithm: 遗传算法主类，包含选择、交叉、变异等操作
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from torch_geometric.data import Data


@dataclass
class Individual:
    """
    遗传算法个体（染色体），表示一个 VN 到 SN 的映射方案。
    
    属性:
        mapping: VN节点索引到SN节点ID的映射字典 {vn_idx: sn_id}
        fitness: 适应度值（越大越好）
        is_feasible: 是否满足资源约束
    """
    mapping: Dict[int, int]
    fitness: float = -float("inf")
    is_feasible: bool = False

    def __len__(self) -> int:
        """返回映射的节点数量"""
        return len(self.mapping)


class GeneticAlgorithm:
    """
    遗传算法主类，用于求解 VNE 问题。
    
    算法流程：
    1. 初始化种群（随机生成映射）
    2. 评估适应度
    3. 选择、交叉、变异
    4. 迭代直到满足终止条件
    """

    def __init__(
        self,
        vn: Data,
        sn_graph: nx.Graph,
        sn_node_list: List[int],
        sn_max_capacity: Dict[str, float],
        *,
        non_constraint_vn_indices: Optional[List[int]] = None,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 5,
        tournament_size: int = 3,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        初始化遗传算法。
        
        参数:
            vn: 虚拟网络图数据
            sn_graph: 底层网络图（NetworkX Graph）
            sn_node_list: SN节点ID列表（有序）
            sn_max_capacity: SN最大容量字典（用于归一化）
            non_constraint_vn_indices: 非约束节点的VN索引列表（None表示所有节点都是非约束节点）
            population_size: 种群大小
            max_generations: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_size: 精英个体数量（直接保留到下一代）
            tournament_size: 锦标赛选择的大小
            seed: 随机种子
            verbose: 是否打印详细信息
        """
        self.vn = vn
        self.sn_graph = sn_graph
        self.sn_node_list = sn_node_list
        self.sn_max_capacity = sn_max_capacity
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.verbose = verbose

        # 设置随机种子
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
        else:
            random.seed(42)
            np.random.seed(42)

        # VN节点数量
        self.num_vn_nodes = int(vn.x.size(0))
        self.num_sn_nodes = len(sn_node_list)
        
        # 非约束节点列表（如果为None，所有节点都是非约束节点）
        if non_constraint_vn_indices is None:
            self.non_constraint_vn_indices = list(range(self.num_vn_nodes))
        else:
            self.non_constraint_vn_indices = non_constraint_vn_indices
        self.num_non_constraint = len(self.non_constraint_vn_indices)

        # 计算VN节点的资源需求（绝对值，只计算非约束节点）
        self.vn_demands: List[Tuple[float, float, float]] = []
        for i in range(self.num_vn_nodes):
            feats = vn.x[i]
            cpu = float(feats[0].item()) * (sn_max_capacity["cpu_max"] + 1e-8)
            mem = float(feats[1].item()) * (sn_max_capacity["mem_max"] + 1e-8)
            disk = float(feats[2].item()) * (sn_max_capacity["disk_max"] + 1e-8)
            self.vn_demands.append((cpu, mem, disk))

    def _generate_random_individual(self, sn_resources: Dict[int, Dict[str, float]]) -> Individual:
        """
        生成一个随机个体（随机映射，只对非约束节点）。
        
        参数:
            sn_resources: SN节点当前剩余资源字典 {sn_id: {'cpu_res': ..., 'mem_res': ..., 'disk_res': ...}}
        
        返回:
            随机生成的个体（只包含非约束节点的映射）
        """
        mapping: Dict[int, int] = {}
        
        # 只对非约束节点，随机选择一个满足资源约束的SN节点
        for vn_idx in self.non_constraint_vn_indices:
            cpu_need, mem_need, disk_need = self.vn_demands[vn_idx]
            candidates = []
            
            for sn_id in self.sn_node_list:
                res = sn_resources[sn_id]
                if (
                    res["cpu_res"] + 1e-9 >= cpu_need
                    and res["mem_res"] + 1e-9 >= mem_need
                    and res["disk_res"] + 1e-9 >= disk_need
                ):
                    candidates.append(sn_id)
            
            if candidates:
                mapping[vn_idx] = random.choice(candidates)
            else:
                # 如果找不到满足约束的节点，随机选择一个（后续适应度评估会惩罚）
                mapping[vn_idx] = random.choice(self.sn_node_list)
        
        return Individual(mapping=mapping)

    def check_feasibility(
        self, mapping: Dict[int, int], sn_resources: Dict[int, Dict[str, float]]
    ) -> bool:
        """
        检查映射是否满足资源约束。
        
        参数:
            mapping: VN到SN的映射
            sn_resources: SN节点剩余资源
        
        返回:
            是否满足约束
        """
        # 创建资源使用计数器
        resource_usage: Dict[int, Dict[str, float]] = {}
        for sn_id in self.sn_node_list:
            resource_usage[sn_id] = {"cpu": 0.0, "mem": 0.0, "disk": 0.0}
        
        # 累计资源需求
        for vn_idx, sn_id in mapping.items():
            cpu_need, mem_need, disk_need = self.vn_demands[vn_idx]
            resource_usage[sn_id]["cpu"] += cpu_need
            resource_usage[sn_id]["mem"] += mem_need
            resource_usage[sn_id]["disk"] += disk_need
        
        # 检查是否超出资源限制
        for sn_id in self.sn_node_list:
            usage = resource_usage[sn_id]
            available = sn_resources[sn_id]
            if (
                usage["cpu"] > available["cpu_res"] + 1e-9
                or usage["mem"] > available["mem_res"] + 1e-9
                or usage["disk"] > available["disk_res"] + 1e-9
            ):
                return False
        
        return True

    def compute_path_length(self, mapping: Dict[int, int]) -> float:
        """
        计算映射的路径总长度（跳数）。
        
        参数:
            mapping: VN到SN的映射
        
        返回:
            所有VN链路对应的SN路径总跳数
        """
        total_hops = 0.0
        edge_index = self.vn.edge_index
        
        for i in range(edge_index.size(1)):
            u = int(edge_index[0, i].item())
            v = int(edge_index[1, i].item())
            sn_u = mapping.get(u)
            sn_v = mapping.get(v)
            
            if sn_u is None or sn_v is None:
                continue
            
            if sn_u == sn_v:
                # 同一节点，跳数为0
                continue
            
            try:
                path = nx.shortest_path(self.sn_graph, source=sn_u, target=sn_v)
                total_hops += len(path) - 1  # 跳数 = 路径长度 - 1
            except nx.NetworkXNoPath:
                # 路径不存在，给予大惩罚
                total_hops += 1000.0
        
        return total_hops

    def _evaluate_fitness(
        self, individual: Individual, sn_resources: Dict[int, Dict[str, float]]
    ) -> float:
        """
        评估个体的适应度。
        
        适应度函数设计：
        - 如果不可行（资源约束不满足），给予大惩罚
        - 如果可行，考虑路径长度和资源利用率
        
        参数:
            individual: 待评估的个体
            sn_resources: SN节点剩余资源
        
        返回:
            适应度值（越大越好）
        """
        # 检查可行性
        is_feasible = self.check_feasibility(individual.mapping, sn_resources)
        individual.is_feasible = is_feasible
        
        if not is_feasible:
            # 不可行个体给予大惩罚
            individual.fitness = -10000.0
            return individual.fitness
        
        # 可行个体：计算路径长度
        path_length = self.compute_path_length(individual.mapping)
        
        # 计算资源利用率（资源使用量 / 总容量）
        total_used = {"cpu": 0.0, "mem": 0.0, "disk": 0.0}
        total_available = {"cpu": 0.0, "mem": 0.0, "disk": 0.0}
        
        for sn_id in self.sn_node_list:
            res = sn_resources[sn_id]
            total_available["cpu"] += res["cpu_res"]
            total_available["mem"] += res["mem_res"]
            total_available["disk"] += res["disk_res"]
        
        resource_usage: Dict[int, Dict[str, float]] = {}
        for sn_id in self.sn_node_list:
            resource_usage[sn_id] = {"cpu": 0.0, "mem": 0.0, "disk": 0.0}
        
        for vn_idx, sn_id in individual.mapping.items():
            cpu_need, mem_need, disk_need = self.vn_demands[vn_idx]
            resource_usage[sn_id]["cpu"] += cpu_need
            resource_usage[sn_id]["mem"] += mem_need
            resource_usage[sn_id]["disk"] += disk_need
        
        for sn_id in self.sn_node_list:
            usage = resource_usage[sn_id]
            total_used["cpu"] += usage["cpu"]
            total_used["mem"] += usage["mem"]
            total_used["disk"] += usage["disk"]
        
        # 资源利用率（归一化）
        cpu_util = (
            total_used["cpu"] / (total_available["cpu"] + 1e-8)
            if total_available["cpu"] > 0
            else 0.0
        )
        mem_util = (
            total_used["mem"] / (total_available["mem"] + 1e-8)
            if total_available["mem"] > 0
            else 0.0
        )
        disk_util = (
            total_used["disk"] / (total_available["disk"] + 1e-8)
            if total_available["disk"] > 0
            else 0.0
        )
        avg_util = (cpu_util + mem_util + disk_util) / 3.0
        
        # 适应度 = 资源利用率 - 路径长度惩罚（归一化）
        # 路径长度归一化：除以最大可能跳数（假设最坏情况）
        max_possible_hops = self.num_vn_nodes * (self.num_sn_nodes - 1)
        normalized_hops = path_length / (max_possible_hops + 1e-8)
        
        # 适应度 = 资源利用率权重 - 路径长度权重
        fitness = 10.0 * avg_util - 5.0 * normalized_hops
        
        individual.fitness = fitness
        return fitness

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """
        锦标赛选择：从种群中随机选择 tournament_size 个个体，返回适应度最高的。
        
        参数:
            population: 当前种群
        
        返回:
            选中的个体
        """
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        交叉操作：部分映射交换（只对非约束节点）。
        
        参数:
            parent1, parent2: 两个父代个体
        
        返回:
            两个子代个体
        """
        if random.random() > self.crossover_rate:
            # 不进行交叉，直接返回父代
            return Individual(mapping=parent1.mapping.copy()), Individual(
                mapping=parent2.mapping.copy()
            )
        
        # 随机选择一部分非约束VN节点进行交换
        if self.num_non_constraint > 0:
            num_crossover = random.randint(1, max(1, self.num_non_constraint // 2))
            crossover_points = random.sample(self.non_constraint_vn_indices, k=num_crossover)
        else:
            crossover_points = []
        
        child1_mapping = parent1.mapping.copy()
        child2_mapping = parent2.mapping.copy()
        
        for vn_idx in crossover_points:
            if vn_idx in parent2.mapping:
                child1_mapping[vn_idx] = parent2.mapping[vn_idx]
            if vn_idx in parent1.mapping:
                child2_mapping[vn_idx] = parent1.mapping[vn_idx]
        
        return Individual(mapping=child1_mapping), Individual(mapping=child2_mapping)

    def _mutate(
        self, individual: Individual, sn_resources: Dict[int, Dict[str, float]]
    ) -> Individual:
        """
        变异操作：随机改变部分映射（只对非约束节点）。
        
        参数:
            individual: 待变异的个体
            sn_resources: SN节点剩余资源
        
        返回:
            变异后的个体
        """
        if random.random() > self.mutation_rate:
            return Individual(mapping=individual.mapping.copy())
        
        mutated_mapping = individual.mapping.copy()
        
        # 随机选择一部分非约束VN节点进行变异
        if self.num_non_constraint > 0:
            num_mutations = random.randint(1, max(1, self.num_non_constraint // 5))
            mutation_points = random.sample(self.non_constraint_vn_indices, k=num_mutations)
        else:
            mutation_points = []
        
        for vn_idx in mutation_points:
            cpu_need, mem_need, disk_need = self.vn_demands[vn_idx]
            candidates = []
            
            for sn_id in self.sn_node_list:
                res = sn_resources[sn_id]
                if (
                    res["cpu_res"] + 1e-9 >= cpu_need
                    and res["mem_res"] + 1e-9 >= mem_need
                    and res["disk_res"] + 1e-9 >= disk_need
                ):
                    candidates.append(sn_id)
            
            if candidates:
                mutated_mapping[vn_idx] = random.choice(candidates)
            else:
                # 如果找不到满足约束的，随机选择一个
                mutated_mapping[vn_idx] = random.choice(self.sn_node_list)
        
        return Individual(mapping=mutated_mapping)

    def evolve(
        self, sn_resources: Dict[int, Dict[str, float]]
    ) -> Tuple[Dict[int, int], float]:
        """
        执行遗传算法进化过程。
        
        参数:
            sn_resources: SN节点当前剩余资源字典
        
        返回:
            (best_mapping, best_fitness): 最佳映射和适应度
        """
        # 1. 初始化种群
        population: List[Individual] = []
        for _ in range(self.population_size):
            ind = self._generate_random_individual(sn_resources)
            self._evaluate_fitness(ind, sn_resources)
            population.append(ind)
        
        # 记录最佳个体
        best_individual = max(population, key=lambda ind: ind.fitness)
        
        if self.verbose:
            print(
                f"[GA] 初始种群: 最佳适应度={best_individual.fitness:.4f}, "
                f"可行个体数={sum(1 for ind in population if ind.is_feasible)}/{self.population_size}"
            )
        
        # 2. 进化循环
        for generation in range(self.max_generations):
            # 评估所有个体
            for individual in population:
                self._evaluate_fitness(individual, sn_resources)
            
            # 更新最佳个体
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = Individual(
                    mapping=current_best.mapping.copy(),
                    fitness=current_best.fitness,
                    is_feasible=current_best.is_feasible,
                )
            
            # 按适应度排序
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # 3. 生成新种群
            new_population: List[Individual] = []
            
            # 保留精英个体
            for i in range(min(self.elite_size, len(population))):
                new_population.append(
                    Individual(mapping=population[i].mapping.copy())
                )
            
            # 生成剩余个体（通过选择、交叉、变异）
            while len(new_population) < self.population_size:
                # 选择
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1, sn_resources)
                child2 = self._mutate(child2, sn_resources)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
            
            if self.verbose and (generation + 1) % 20 == 0:
                feasible_count = sum(1 for ind in population if ind.is_feasible)
                print(
                    f"[GA] 第 {generation+1}/{self.max_generations} 代: "
                    f"最佳适应度={best_individual.fitness:.4f}, "
                    f"可行个体数={feasible_count}/{self.population_size}"
                )
        
        # 最终评估
        for individual in population:
            self._evaluate_fitness(individual, sn_resources)
        
        final_best = max(population, key=lambda ind: ind.fitness)
        if final_best.fitness > best_individual.fitness:
            best_individual = Individual(
                mapping=final_best.mapping.copy(),
                fitness=final_best.fitness,
                is_feasible=final_best.is_feasible,
            )
        
        if self.verbose:
            print(
                f"[GA] 进化完成: 最佳适应度={best_individual.fitness:.4f}, "
                f"可行={best_individual.is_feasible}"
            )
        
        return best_individual.mapping, best_individual.fitness

