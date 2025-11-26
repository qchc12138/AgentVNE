"""
遗传算法模块：用于解决 VNE（Virtual Network Embedding）问题。

主要组件：
- ga_core: 遗传算法核心实现（Individual, GeneticAlgorithm）
"""

from genetic_algorithm.ga_core import GeneticAlgorithm, Individual

__all__ = ["GeneticAlgorithm", "Individual"]

