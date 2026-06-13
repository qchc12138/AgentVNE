#!/usr/bin/env python3
"""Genetic Algorithm baseline for optimal VN embedding via population-based search."""

import random
from typing import Dict, List, Optional


class GA:
    """Genetic Algorithm with population-based search for optimal VN embedding."""

    def __init__(self, pop_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, elite_count: int = 5):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count

    def __call__(self, env) -> Optional[Dict[int, int]]:
        vn = env.current_vn
        if vn is None:
            return None

        vn_nodes = vn["topo"]["nodes"]
        N_vn = len(vn_nodes)
        N_sn = env.num_sn_nodes
        vn_links = vn["topo"].get("links", [])

        legal_per_vn = {}
        for vi in range(N_vn):
            mask = env.get_legal_mask(vi)
            legal_per_vn[vi] = [j for j, ok in enumerate(mask) if ok]
            if not legal_per_vn[vi]:
                return None

        def _fitness(individual: List[int]) -> float:
            temp_res = {
                j: {"cpu": env.sn_nodes[j].get("cpu", 0.0),
                    "memory": env.sn_nodes[j].get("memory", 0.0),
                    "disk": env.sn_nodes[j].get("disk", 0.0)}
                for j in range(N_sn)
            }
            for vi, sj in enumerate(individual):
                vn_node = vn_nodes[vi]
                r = temp_res[sj]
                demand_cpu = float(vn_node.get("cpu", 0))
                demand_mem = float(vn_node.get("memory", 0))
                demand_disk = float(vn_node.get("disk", 0))
                if demand_cpu > r["cpu"] + 1e-9 or demand_mem > r["memory"] + 1e-9 or demand_disk > r["disk"] + 1e-9:
                    return -1000.0
                r["cpu"] -= demand_cpu
                r["memory"] -= demand_mem
                r["disk"] -= demand_disk

            comm_penalty = 0.0
            for link in vn_links:
                vs = int(link["source"])
                vd = int(link["target"])
                if vs >= N_vn or vd >= N_vn:
                    continue
                ss = individual[vs]
                sd = individual[vd]
                try:
                    import networkx as nx
                    path_len = nx.shortest_path_length(env.sn_graph, source=ss, target=sd, weight="weight")
                except Exception:
                    path_len = N_sn
                comm_penalty += path_len
            avg_penalty = comm_penalty / max(len(vn_links), 1)
            return env.acceptance_reward - env.comm_delay_weight * avg_penalty

        pop = []
        for _ in range(self.pop_size):
            ind = []
            for vi in range(N_vn):
                ind.append(random.choice(legal_per_vn[vi]))
            pop.append(ind)
        pop_fit = [None] * self.pop_size

        best_ind, best_fit = None, -1e12

        for gen in range(self.generations):
            for i, ind in enumerate(pop):
                pop_fit[i] = _fitness(ind)
                if pop_fit[i] > best_fit:
                    best_fit = pop_fit[i]
                    best_ind = ind.copy()

            ranked = sorted(range(self.pop_size), key=lambda i: pop_fit[i], reverse=True)
            new_pop = [pop[ranked[i]].copy() for i in range(self.elite_count)]

            while len(new_pop) < self.pop_size:
                t1 = random.choices(ranked[:max(1, self.pop_size // 2)], k=3)
                t2 = random.choices(ranked[:max(1, self.pop_size // 2)], k=3)
                p1 = pop[max(t1, key=lambda i: pop_fit[i])]
                p2 = pop[max(t2, key=lambda i: pop_fit[i])]
                if N_vn < 2:
                    child = p1.copy()
                else:
                    cp1, cp2 = sorted(random.sample(range(N_vn), 2))
                    child = p1[:cp1] + p2[cp1:cp2] + p1[cp2:]
                for vi in range(N_vn):
                    if random.random() < self.mutation_rate:
                        child[vi] = random.choice(legal_per_vn[vi])
                new_pop.append(child)

            pop = new_pop

        if best_ind is None:
            return None
        return {vi: sj for vi, sj in enumerate(best_ind)}
