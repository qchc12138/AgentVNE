#!/usr/bin/env python3
"""Greedy-NodeRank baseline: priority-based placement using precomputed NodeRank scores."""

import json
from typing import Dict, Optional


class GreedyNodeRank:
    """Greedy placement using NodeRank scores from precomputed topo JSON.

    Supports two JSON formats:
    - SN format with ``node_details``: list of {node_id, noderank} objects
    - VN format with flat ``noderank``: list of scores indexed by node position
    """

    def __init__(self, sn_noderank_path: str, vn_noderank_path: str):
        self.sn_nr = self._load_sn(sn_noderank_path)
        self.vn_nr = self._load_vn(vn_noderank_path)

    @staticmethod
    def _load_sn(path: str) -> Dict[int, float]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        details = data.get("node_details")
        if details:
            return {d["node_id"]: d["noderank"] for d in details}
        nr = data.get("noderank", [])
        return {i: v for i, v in enumerate(nr)}

    @staticmethod
    def _load_vn(path: str) -> Dict[int, float]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        details = data.get("node_details")
        if details:
            return {d["node_id"]: d["noderank"] for d in details}
        nr = data.get("noderank", [])
        return {i: v for i, v in enumerate(nr)}

    def __call__(self, env) -> Optional[Dict[int, int]]:
        vn = env.current_vn
        if vn is None:
            return None
        vn_nodes = vn["topo"]["nodes"]
        vn_order = sorted(
            range(len(vn_nodes)),
            key=lambda i: self.vn_nr.get(vn_nodes[i].get("id", i), 0.0),
            reverse=True,
        )
        mapping: Dict[int, int] = {}
        temp_mapping: Dict[int, int] = {}
        for vn_idx in vn_order:
            mask = env.get_legal_mask(vn_idx)
            candidates = []
            for sn_idx in range(env.num_sn_nodes):
                if mask[sn_idx] and env.is_valid_assignment(vn_idx, sn_idx, temp_mapping=temp_mapping):
                    sn_id = env._sn_nodes[sn_idx].get("id", sn_idx)
                    nr = self.sn_nr.get(sn_id, 0.0)
                    candidates.append((sn_idx, nr))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_sn = candidates[0][0]
            mapping[vn_idx] = best_sn
            temp_mapping[vn_idx] = best_sn
        return mapping
