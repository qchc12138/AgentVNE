#!/usr/bin/env python3
"""GRC (Graph Convolution Ranking) baseline: frozen GCN-only model for placement."""

from typing import Dict, Optional

import torch


def _load_checkpoint_config(checkpoint, model_state_dict):
    """Extract model config from checkpoint, with fallback to weight inspection."""
    config = checkpoint.get('model_config', {})
    if 'num_nodes_j' not in config and 'ntn.W' in model_state_dict:
        ntn_shape = model_state_dict['ntn.W'].shape
        config = {'num_nodes_j': ntn_shape[0]}
    return config


class GRC:
    """GCN-based similarity ranking: uses frozen pretrained model for heuristic placement.

    Described in the paper as Graph Convolution Ranking (GRC).
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        from model import SimuVNE
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        config = _load_checkpoint_config(ckpt, state)
        input_dim = config.get("input_dim", 6)
        hidden_dim = config.get("hidden_dim", 64)
        num_nodes_j = config.get("num_nodes_j", 7)
        hist_dim = config.get("hist_dim", 32)

        self.model = SimuVNE(input_dim=input_dim, hidden_dim=hidden_dim,
                              hist_dim=hist_dim, num_nodes_j=num_nodes_j)
        self.model.load_state_dict(state, strict=False)
        self.model.to(device)
        self.model.eval()

    def __call__(self, env) -> Optional[Dict[int, int]]:
        sn_data, vn_data = env.get_state()
        if vn_data is None:
            return None
        with torch.no_grad():
            probs = self.model(vn_data.to(self.device), sn_data.to(self.device))
        return env._greedy_place(probs)
