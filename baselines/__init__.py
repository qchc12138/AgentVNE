#!/usr/bin/env python3
"""Baseline strategies for Virtual Network Embedding.

Provides:
  - GreedySN: greedy placement by highest residual resource
  - GreedyNodeRank: greedy placement using precomputed NodeRank scores
  - GA: genetic algorithm with population-based search
  - GRC: GCN-based similarity ranking
  - PretrainStrategy: frozen pretrained SimuVNE model inference
  - AgentVNEStrategy: post-PPO-finetuned model inference
"""
from typing import Any, Callable, Dict

from baselines.greedy import GreedySN
from baselines.noderank import GreedyNodeRank
from baselines.ga import GA


STRATEGY_REGISTRY: Dict[str, type] = {
    "greedy": GreedySN,
    "greedy-sn": GreedySN,
    "greedy_noderank": GreedyNodeRank,
    "noderank": GreedyNodeRank,
    "ga": GA,
}


def resolve_strategy(name: str, extra_args: Dict[str, Any]) -> Callable:
    """Instantiate a named strategy with its constructor args.

    Torch-dependent strategies (grc, pretrain, agentvne) are imported
    lazily so the package remains importable without PyTorch.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        # lazy imports for torch-dependent strategies
        if name in ("grc",):
            from baselines.grc import GRC as _GRC
            cls = _GRC
            STRATEGY_REGISTRY["grc"] = cls
        elif name in ("pretrain", "pretrained"):
            from baselines.model_based import PretrainStrategy as _PS
            cls = _PS
            STRATEGY_REGISTRY["pretrain"] = cls
            STRATEGY_REGISTRY["pretrained"] = cls
        elif name in ("agentvne", "finetuned"):
            from baselines.model_based import AgentVNEStrategy as _AV
            cls = _AV
            STRATEGY_REGISTRY["agentvne"] = cls
            STRATEGY_REGISTRY["finetuned"] = cls
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    sig_params: Dict[str, list] = {
        "greedy": [],
        "greedy-sn": [],
        "greedy_noderank": ["sn_noderank_path", "vn_noderank_path"],
        "noderank": ["sn_noderank_path", "vn_noderank_path"],
        "ga": ["pop_size", "generations"],
        "grc": ["model_path", "device"],
        "pretrain": ["model_path", "device"],
        "pretrained": ["model_path", "device"],
        "agentvne": ["finetuned_model_path", "device"],
        "finetuned": ["finetuned_model_path", "device"],
    }.get(name, [])
    kwargs = {k: extra_args[k] for k in sig_params if k in extra_args}
    return cls(**kwargs)
