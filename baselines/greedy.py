#!/usr/bin/env python3
"""Greedy-SN baseline: picks the highest-residual-resource legal SN node per VN node."""

from typing import Dict, List, Optional


def _resource_score(sn_node: Dict) -> float:
    """Normalised remaining resource score."""
    return float(sn_node.get("cpu", 0)) + float(sn_node.get("memory", 0)) + float(sn_node.get("disk", 0))


def _sort_vn_by_demand(env) -> List[int]:
    """Return VN node indices sorted by demand descending (constrained first)."""
    vn_nodes = env.current_vn["topo"]["nodes"]
    constrained = [i for i, n in enumerate(vn_nodes) if n.get("constraint_node") is not None]
    unconstrained = [i for i in range(len(vn_nodes)) if i not in constrained]
    unconstrained.sort(
        key=lambda i: float(vn_nodes[i].get("cpu", 0)) + float(vn_nodes[i].get("memory", 0)) + float(vn_nodes[i].get("disk", 0)),
        reverse=True,
    )
    return constrained + unconstrained


class GreedySN:
    """Greedy placement: for each VN node, pick the highest-resource legal SN node."""

    def __init__(self):
        pass

    def __call__(self, env) -> Optional[Dict[int, int]]:
        vn = env.current_vn
        if vn is None:
            return None
        vn_order = _sort_vn_by_demand(env)
        mapping: Dict[int, int] = {}
        temp_mapping: Dict[int, int] = {}
        for vn_idx in vn_order:
            mask = env.get_legal_mask(vn_idx)
            best_sn, best_score = None, -1.0
            for sn_idx in range(env.num_sn_nodes):
                if not mask[sn_idx]:
                    continue
                if not env.is_valid_assignment(vn_idx, sn_idx, temp_mapping=temp_mapping):
                    continue
                score = _resource_score(env.sn_nodes[sn_idx])
                if score > best_score:
                    best_score = score
                    best_sn = sn_idx
            if best_sn is None:
                return None
            mapping[vn_idx] = best_sn
            temp_mapping[vn_idx] = best_sn
        return mapping
