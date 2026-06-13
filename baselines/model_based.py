import os
import torch
from typing import Dict, Optional


def _load_checkpoint_config(checkpoint, model_state_dict):
    """Extract model config from checkpoint, with fallback to weight inspection."""
    config = checkpoint.get('model_config', {})
    if 'num_nodes_j' not in config and 'ntn.W' in model_state_dict:
        ntn_shape = model_state_dict['ntn.W'].shape
        config = {'num_nodes_j': ntn_shape[0]}
    return config


class PretrainStrategy:
    """Pretrained SimuVNE model for inference (no PPO finetuning)."""

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


class AgentVNEStrategy:
    """Post-PPO-finetuned SimuVNE model for inference.

    Loads from the finetuned model path (policy_network_latest.pth) produced
    by fine_tuning.py.
    """

    def __init__(self, finetuned_model_path: str, device: str = "cpu"):
        self.device = device
        from model import SimuVNE

        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(
                f"Finetuned model not found at {finetuned_model_path}. "
                f"Run fine_tuning.py first to produce this checkpoint."
            )

        ckpt = torch.load(finetuned_model_path, map_location=device, weights_only=True)
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