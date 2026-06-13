#!/usr/bin/env python3
"""
AgentVNE Evaluation Framework - Phase 4

Implements:
  Baselines: Greedy-SN, Greedy-NodeRank, GA, GRC
  Model-based: Pretrain (ft_n), AgentVNE (ft_1)
  Unified evaluation loop, parameter sweep, plotting (Figure 4/5)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from env import SimuVNEEnv, WorkflowGenerator
from baselines import resolve_strategy, GreedySN, GreedyNodeRank, GA
from baselines.grc import GRC
from baselines.model_based import PretrainStrategy, AgentVNEStrategy

# =========================================================================
#  helpers
# =========================================================================

def _warn(msg: str) -> None:
    print(f"  [!] {msg}", file=sys.stderr)

# =========================================================================
#  Evaluation loop
# =========================================================================

def run_evaluation(
    strategy_fn: Callable,
    sn_topo: str,
    workflow_types: Dict[str, str],
    arrival_rate: float = 0.25,
    mean_lifetime: float = 40.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 11000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one simulation with the given strategy.

    Returns a stats dict with: acceptance_rate, avg_comm_delay, total_reward,
    avg_time_per_vn, total_vn, accepted_vn, rejected_vn.
    """
    wf_gen = WorkflowGenerator(
        arrival_rate=arrival_rate, mean_lifetime=mean_lifetime,
        workflow_types=workflow_types, max_arrived_tasks=max_arrived_tasks, seed=seed)
    env = SimuVNEEnv(sn_topo, wf_gen, seed=seed, max_time_steps=max_time_steps)

    (sn_data, vn_data) = env.reset()
    done = False
    placement_times: List[float] = []
    comm_delays: List[float] = []

    while not done:
        if vn_data is None:
            (sn_data, vn_data), _, done, _ = env.step(torch.zeros(env.num_sn_nodes, env.num_sn_nodes))
            continue

        t0 = time.perf_counter()
        mapping = strategy_fn(env)
        t1 = time.perf_counter()
        placement_times.append(t1 - t0)

        if mapping is None:
            # reject
            (sn_data, vn_data), _, done, _ = env.step(torch.zeros(env.num_sn_nodes, env.num_sn_nodes))
        else:
            comm_penalty = 0.0
            vn_links = env.current_vn["topo"].get("links", [])
            if vn_links:
                for link in vn_links:
                    vs = int(link["source"])
                    vd = int(link["target"])
                    if vs in mapping and vd in mapping:
                        try:
                            import networkx as nx
                            pl = nx.shortest_path_length(env.sn_graph, source=mapping[vs], target=mapping[vd], weight="weight")
                            comm_penalty += pl
                        except Exception:
                            comm_penalty += env.num_sn_nodes
                comm_delays.append(comm_penalty / len(vn_links))

            # execute placement via env.step with a dummy action (mapping already decided)
            N_vn = len(mapping)
            action = torch.zeros(N_vn, env.num_sn_nodes)
            for vi, sj in mapping.items():
                action[vi, sj] = 1.0
            (sn_data, vn_data), _, done, _ = env.step(action)

    stats = env.get_stats()
    stats["avg_comm_delay"] = np.mean(comm_delays) if comm_delays else 0.0
    stats["avg_time_per_vn_ms"] = np.mean(placement_times) * 1000.0 if placement_times else 0.0
    stats["total_vn"] = stats["total_arrived"]
    stats["accepted_vn"] = stats["total_accepted"]
    stats["rejected_vn"] = stats["total_arrived"] - stats["total_accepted"]

    if verbose:
        print(f"  acceptance={stats['acceptance_rate']:.2%}  "
              f"comm_delay={stats['avg_comm_delay']:.2f}  "
              f"time/VN={stats['avg_time_per_vn_ms']:.1f}ms  "
              f"({stats['accepted_vn']}/{stats['total_vn']})")

    return stats


# =========================================================================
#  Parameter sweep
# =========================================================================

def parameter_sweep(
    strategies: Dict[str, Callable],
    sn_topo: str,
    workflow_types: Dict[str, str],
    arrival_rates: List[float],
    mean_lifetimes: List[float],
    max_time_steps: int = 11000,
    max_arrived_tasks: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict]:
    """Run evaluation across strategies and parameter combinations."""
    results = []
    for sname, sfn in strategies.items():
        for ar in arrival_rates:
            for lt in mean_lifetimes:
                print(f"\n[{sname}] arrival_rate={ar} mean_lifetime={lt}")
                stats = run_evaluation(
                    strategy_fn=sfn,
                    sn_topo=sn_topo,
                    workflow_types=workflow_types,
                    arrival_rate=ar,
                    mean_lifetime=lt,
                    max_arrived_tasks=max_arrived_tasks,
                    max_time_steps=max_time_steps,
                    seed=seed,
                    verbose=verbose,
                )
                stats["strategy"] = sname
                stats["arrival_rate"] = ar
                stats["mean_lifetime"] = lt
                results.append(stats)
    return results


# =========================================================================
#  Plotting
# =========================================================================

def plot_results(results: List[Dict], output_dir: str = ".") -> None:
    """Generate Figure 4 (acceptance rate vs arrival rate) and
    Figure 5 (communication delay bar chart)."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Figure 4: Acceptance Rate vs Arrival Rate ----
    by_strat = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_strat[r["strategy"]][r["arrival_rate"]].append(r["acceptance_rate"])

    strat_names = sorted(by_strat.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(strat_names)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, sname in enumerate(strat_names):
        ars = sorted(by_strat[sname].keys())
        vals = [np.mean(by_strat[sname][a]) for a in ars]
        ax.plot(ars, vals, "o-", color=colors[idx], label=sname, linewidth=2, markersize=6)
    ax.set_xlabel("Arrival Rate", fontsize=12)
    ax.set_ylabel("Acceptance Rate", fontsize=12)
    ax.set_title("Acceptance Rate vs Arrival Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(output_dir, "figure4_acceptance_rate.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved {path1}")
    plt.close(fig)

    # ---- Figure 5: Communication Delay ----
    by_strat_delay = defaultdict(list)
    for r in results:
        by_strat_delay[r["strategy"]].append(r.get("avg_comm_delay", 0.0))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(strat_names))
    means = [np.mean(by_strat_delay[s]) for s in strat_names]
    stds = [np.std(by_strat_delay[s]) for s in strat_names]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names, fontsize=10)
    ax.set_ylabel("Avg Communication Delay (weighted distance)", fontsize=12)
    ax.set_title("Communication Delay by Strategy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    path2 = os.path.join(output_dir, "figure5_comm_delay.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved {path2}")
    plt.close(fig)

    # ---- Save raw results JSON ----
    path3 = os.path.join(output_dir, "evaluation_results.json")
    with open(path3, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved {path3}")


# =========================================================================
#  CLI
# =========================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="AgentVNE Evaluation Suite")
    parser.add_argument("--sn_topology", default=os.path.join(script_dir, "topo", "SN_topology.json"))
    parser.add_argument("--sn_noderank", default=os.path.join(script_dir, "topo", "SN_topology_noderank.json"))
    parser.add_argument("--vn_noderank", default=os.path.join(script_dir, "Workflow_topo", "workflow1_noderank.json"))
    parser.add_argument("--workflow", action="append",
                        help="Format: name=path  e.g. wf1=Workflow_topo/workflow1_topo.json",
                        default=["workflow1=" + os.path.join(script_dir, "Workflow_topo", "workflow1_topo.json")])
    parser.add_argument("--strategies", nargs="+", default=["greedy", "ga", "pretrain"],
                        help="Strategy names: greedy, noderank, ga, grc, pretrain, agentvne")
    parser.add_argument("--pretrain_model", default=os.path.join(script_dir, "pretrain_outputs", "checkpoint_latest.pt"))
    parser.add_argument("--finetuned_model", default=os.path.join(script_dir, "finetuning_output", "policy_network_latest.pth"))
    parser.add_argument("--parameter", action="append", default=[],
                        help="Override default params: arrival_rate=0.25,mean_lifetime=40,...")
    parser.add_argument("--arrival_rates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--mean_lifetimes", nargs="+", type=float, default=[20.0, 40.0, 60.0])
    parser.add_argument("--max_time_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=os.path.join(script_dir, "eval_output"))
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Generate comparison plots")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # parse workflow types
    workflow_types: Dict[str, str] = {}
    for wf_spec in args.workflow:
        if "=" in wf_spec:
            name, path = wf_spec.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(wf_spec))[0]
            path = wf_spec
        if not os.path.isabs(path):
            path = os.path.join(script_dir, path)
        workflow_types[name] = path

    # parse --parameter overrides
    param_overrides: Dict[str, Any] = {}
    for p in args.parameter:
        for kv in p.split(","):
            k, v = kv.split("=", 1)
            try:
                param_overrides[k] = float(v)
            except ValueError:
                param_overrides[k] = v

    arrival_rates = param_overrides.get("arrival_rates", args.arrival_rates)
    if not isinstance(arrival_rates, list):
        arrival_rates = [float(arrival_rates)]
    mean_lifetimes = param_overrides.get("mean_lifetimes", args.mean_lifetimes)
    if not isinstance(mean_lifetimes, list):
        mean_lifetimes = [float(mean_lifetimes)]

    # collect extra strategy args
    extra = {
        "device": args.device,
        "model_path": args.pretrain_model,
        "finetuned_model_path": args.finetuned_model,
        "sn_noderank_path": args.sn_noderank,
        "vn_noderank_path": args.vn_noderank,
        "pop_size": 50,
        "generations": 100,
    }
    # Read SN topology to pass num_sn_nodes for model-based strategies
    with open(args.sn_topology, "r", encoding="utf-8") as _f:
        _sn = json.load(_f)
    extra["num_sn_nodes"] = len(_sn.get("nodes", []))

    # resolve strategies
    strategies: Dict[str, Callable] = {}
    for sname in args.strategies:
        try:
            sfn = resolve_strategy(sname, extra)
            strategies[sname] = sfn
        except Exception as e:
            _warn(f"{sname}: {e}")

    if not strategies:
        print("No valid strategies specified. Exiting.")
        return

    print("=" * 60)
    print("AgentVNE Evaluation Framework")
    print(f"  SN topology:    {args.sn_topology}")
    print(f"  Workflows:      {list(workflow_types.keys())}")
    print(f"  Strategies:     {sorted(strategies.keys())}")
    print(f"  Arrival rates:  {arrival_rates}")
    print(f"  Mean lifetimes: {mean_lifetimes}")
    print(f"  Max time steps: {args.max_time_steps}")
    print("=" * 60)

    results = parameter_sweep(
        strategies=strategies,
        sn_topo=args.sn_topology,
        workflow_types=workflow_types,
        arrival_rates=arrival_rates,
        mean_lifetimes=mean_lifetimes,
        max_time_steps=args.max_time_steps,
        max_arrived_tasks=param_overrides.get("max_arrived_tasks", 20),
        seed=args.seed,
        verbose=True,
    )

    if args.plot:
        plot_results(results, output_dir=args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    # quick self-test when run directly
    if len(sys.argv) == 1:
        print("=== Tester Quick Self-Test ===")
        main_args = argparse.Namespace(
            sn_topology=os.path.join(os.path.dirname(os.path.abspath(__file__)), "topo", "SN_topology.json"),
            sn_noderank=os.path.join(os.path.dirname(os.path.abspath(__file__)), "topo", "SN_topology_noderank.json"),
            vn_noderank=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Workflow_topo", "workflow1_noderank.json"),
            workflow=["workflow1=" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "Workflow_topo", "workflow1_topo.json")],
            strategies=["greedy"],
            pretrain_model=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_outputs", "checkpoint_latest.pt"),
            finetuned_model=os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning_output", "policy_network_latest.pth"),
            parameter=[],
            arrival_rates=[0.1],
            mean_lifetimes=[40.0],
            max_time_steps=200,
            seed=42,
            output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output"),
            plot=True,
            device="cpu",
        )
        # run quick test with greedy only (no GPU model needed)
        args = main_args

        workflow_types: Dict[str, str] = {}
        for wf_spec in args.workflow:
            name, path = wf_spec.split("=", 1)
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            workflow_types[name] = path

        extra = {
            "device": "cpu",
            "model_path": args.pretrain_model,
        "finetuned_model_path": args.finetuned_model,
            "sn_noderank_path": args.sn_noderank,
            "vn_noderank_path": args.vn_noderank,
            "pop_size": 50,
            "generations": 100,
        }
        strategies = {}
        for sname in args.strategies:
            try:
                strategies[sname] = resolve_strategy(sname, extra)
            except Exception:
                pass

        results = parameter_sweep(
            strategies=strategies,
            sn_topo=args.sn_topology,
            workflow_types=workflow_types,
            arrival_rates=args.arrival_rates,
            mean_lifetimes=args.mean_lifetimes,
            max_time_steps=args.max_time_steps,
            max_arrived_tasks=20,
            seed=args.seed,
            verbose=True,
        )
        if args.plot:
            plot_results(results, output_dir=args.output_dir)
        print("\n=== Self-test complete ===")
    else:
        main()
