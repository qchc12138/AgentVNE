#!/usr/bin/env python3
"""
AgentVNE Evaluation Framework -- Paper Figure 5 & 6 Reproduction

Figure 5: Communication Hops (line plots)
   (a) Stability over time
   (b) Hops vs arrival rate  [0.05-0.50]
   (c) Hops vs mean lifetime [15-40]

Figure 6: Acceptance Rate (a: line, b/c: bar charts)
   (a) Long-term acceptance over time
   (b) Acceptance vs arrival rate  [0.20, 0.30, 0.40]
   (c) Acceptance vs mean lifetime [20, 30, 40]
"""

import argparse, json, os, sys, time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")
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

def _hops_between(sn_graph, src: int, dst: int, fallback: int = 50) -> int:
    """Return unweighted hop count between two SN nodes."""
    try:
        import networkx as nx
        return nx.shortest_path_length(sn_graph, source=src, target=dst, weight=None)
    except Exception:
        return fallback


# =========================================================================
#  Strategy name mapping (code -> paper label)
# =========================================================================

STRATEGY_LABEL: Dict[str, str] = {
    "greedy":    "Greedy",
    "gal-vne":   "gal-vne",
    "ga":        "GA",
    "grc":       "GRC",
    "pretrain":  "GAL-VNE",
    "agentvne":  "AgentVNE",
}

STRATEGY_COLOR: Dict[str, Tuple[float, float, float]] = {
    "greedy":    "#2ca02c",
    "gal-vne":   "#1f77b4",
    "ga":        "#1f77b4",
    "grc":       "#d62728",
    "pretrain":  "#ff7f0e",
    "agentvne":  "#9467bd",
}

def _clr(name: str) -> Any:
    if name in STRATEGY_COLOR:
        return STRATEGY_COLOR[name]
    idx = sum(ord(c) for c in name) % 10
    return plt.cm.tab10(idx)

def _lbl(name: str) -> str:
    return STRATEGY_LABEL.get(name, name)


# =========================================================================
#  Evaluation loop
# =========================================================================

def run_evaluation(strategy_fn: Callable, sn_topo: str, workflow_types: Dict[str, str],
    arrival_rate: float = 0.25,
    mean_lifetime: float = 40.0,
    max_arrived_tasks: int = 20,
    max_time_steps: int = 11000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one simulation. Returns stats + history / hop_series."""
    wf_gen = WorkflowGenerator(
        arrival_rate=arrival_rate, mean_lifetime=mean_lifetime,
        workflow_types=workflow_types, max_arrived_tasks=max_arrived_tasks, seed=seed)
    env = SimuVNEEnv(sn_topo, wf_gen, seed=seed, max_time_steps=max_time_steps)

    (sn_data, vn_data) = env.reset()
    done, placement_times, comm_delays, accept_history = False, [], [], []
    history: List[Dict[str, Any]] = []
    hop_series: List[Dict[str, Any]] = []

    while not done:
        current_step = env.time_step

        if vn_data is None:
            (sn_data, vn_data), _, done, _ = env.step(torch.zeros(env.num_sn_nodes, env.num_sn_nodes))
            continue

        t0 = time.perf_counter()
        mapping = strategy_fn(env)
        t1 = time.perf_counter()
        placement_times.append(t1 - t0)

        if mapping is None:
            (sn_data, vn_data), _, done, _ = env.step(torch.zeros(env.num_sn_nodes, env.num_sn_nodes))
        else:
            comm_penalty = 0.0
            vn_links = env.current_vn["topo"].get("links", [])
            if vn_links:
                for link in vn_links:
                    vs, vd = int(link["source"]), int(link["target"])
                    if vs in mapping and vd in mapping:
                        comm_penalty += _hops_between(env.sn_graph, mapping[vs], mapping[vd])
                comm_delays.append(comm_penalty / len(vn_links)); accept_history.append(1 if mapping else 0)
            N_vn = len(mapping)
            action = torch.zeros(N_vn, env.num_sn_nodes)
            for vi, sj in mapping.items():
                action[vi, sj] = 1.0
            (sn_data, vn_data), _, done, _ = env.step(action)

        if current_step % 100 == 0 or done:
            s = env.get_stats()
            history.append({
                "time_step": current_step,
                "acceptance_rate": s.get("acceptance_rate", 0.0),
            })
            hop_series.append({
                "time_step": current_step,
                "hops": float(np.mean(comm_delays[-50:])) if comm_delays else 0.0,
            })

    stats = env.get_stats()
    stats["avg_comm_delay"] = np.mean(comm_delays) if comm_delays else 0.0
    stats["avg_time_per_vn_ms"] = np.mean(placement_times) * 1000.0 if placement_times else 0.0
    stats["total_vn"] = stats["total_arrived"]
    stats["accepted_vn"] = stats["total_accepted"]
    stats["rejected_vn"] = stats["total_arrived"] - stats["total_accepted"]

    if verbose:
        print(f"  acceptance={stats['acceptance_rate']:.2%}  "
              f"hops={stats['avg_comm_delay']:.1f}  "
              f"time/VN={stats['avg_time_per_vn_ms']:.1f}ms  "
              f"({stats['accepted_vn']}/{stats['total_vn']})")

    stats["history"] = history
    stats["hop_series"] = hop_series
    return stats


# =========================================================================
#  Sweep helpers
# =========================================================================

def _run_single(strategies, sn_topo, workflow_types, ar, lt, max_ts, max_tasks,
                seed, verbose, sweep_tag=""):
    results: List[Dict] = []
    for sname, sfn in strategies.items():
        if verbose:
            print(f"\n[{sname}] arrival_rate={ar} mean_lifetime={lt} [{sweep_tag}]")
        stats = run_evaluation(sfn, sn_topo, workflow_types, ar, lt,
                               max_tasks, max_ts, seed, verbose)
        stats["strategy"] = sname
        stats["arrival_rate"] = ar
        stats["mean_lifetime"] = lt
        stats["sweep_tag"] = sweep_tag
        results.append(stats)
    return results


def parameter_sweep(strategies, sn_topo, workflow_types,
                    arrival_rates, mean_lifetimes,
                    max_time_steps=11000, max_arrived_tasks=20, seed=42,
                    verbose=True, sweep_mode="grid",
                    fixed_lifetime=None, fixed_arrival=None,
                    sweep_tag=""):
    """Run evaluation across strategies and parameter combinations."""
    results = []
    if sweep_mode in ("arrival", "arrival_rate"):
        lt = fixed_lifetime if fixed_lifetime is not None else mean_lifetimes[0]
        for ar in arrival_rates:
            results.extend(_run_single(strategies, sn_topo, workflow_types, ar, lt,
                                       max_time_steps, max_arrived_tasks, seed,
                                       verbose, sweep_tag))
    elif sweep_mode in ("lifetime", "mean_lifetime"):
        ar = fixed_arrival if fixed_arrival is not None else arrival_rates[0]
        for lt in mean_lifetimes:
            results.extend(_run_single(strategies, sn_topo, workflow_types, ar, lt,
                                       max_time_steps, max_arrived_tasks, seed,
                                       verbose, sweep_tag))
    elif sweep_mode == "stability":
        ar = fixed_arrival if fixed_arrival is not None else arrival_rates[0]
        lt = fixed_lifetime if fixed_lifetime is not None else mean_lifetimes[0]
        results.extend(_run_single(strategies, sn_topo, workflow_types, ar, lt,
                                   max_time_steps, max_arrived_tasks, seed,
                                   verbose, sweep_tag))
    else:
        for ar in arrival_rates:
            for lt in mean_lifetimes:
                results.extend(_run_single(strategies, sn_topo, workflow_types, ar, lt,
                                           max_time_steps, max_arrived_tasks, seed,
                                           verbose, sweep_tag))
    return results


# =========================================================================
#  Plotting  (Paper Figure 5 & 6)
# =========================================================================

def _aggregate(results, strategies, x_field, y_field, tag_filter=None):
    strat_vals = {s: defaultdict(list) for s in strategies}
    for r in results:
        if tag_filter and r.get("sweep_tag", "") != tag_filter:
            continue
        s = r.get("strategy", "")
        if s not in strat_vals:
            continue
        x, y = r.get(x_field), r.get(y_field)
        if x is not None and y is not None:
            strat_vals[s][float(x)].append(float(y))
    out = {}
    for s in strategies:
        d = strat_vals.get(s, {})
        if not d:
            continue
        xs = sorted(d.keys())
        ys = [np.mean(d[x]) for x in xs]
        out[s] = (xs, ys)
    return out


def _plot_lines(ax, data, strategies, marker="o-", ms=5):
    for sname in strategies:
        if sname not in data:
            continue
        xs, ys = data[sname]
        ax.plot(xs, ys, marker, color=_clr(sname), label=_lbl(sname),
                linewidth=2, markersize=ms)


def _plot_bars(ax, data, strategies, bar_width=0.18):
    """Bar chart with grouped bars for each strategy."""
    n_strat = len([s for s in strategies if s in data])
    if n_strat == 0:
        return
    # Get union of all x values
    all_xs = set()
    for s in strategies:
        if s in data:
            all_xs.update(data[s][0])
    xs = sorted(all_xs)
    n_groups = len(xs)
    positions = np.arange(n_groups)
    for i, sname in enumerate(strategies):
        if sname not in data:
            continue
        dx, dy = data[sname]
        # Map each x to a bar position
        bar_vals = []
        for x in xs:
            if x in dx:
                idx = dx.index(x)
                bar_vals.append(dy[idx])
            else:
                bar_vals.append(0)
        offset = (i - (n_strat - 1) / 2) * bar_width
        ax.bar(positions + offset, bar_vals, bar_width,
               color=_clr(sname), label=_lbl(sname), edgecolor='white', linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(x) for x in xs])


def plot_results(results: List[Dict], output_dir: str = ".") -> None:
    """Generate Figure 5 (line) and Figure 6 (a:line, b/c:bar) per paper specs."""
    os.makedirs(output_dir, exist_ok=True)

    strategies = sorted({r.get("strategy", "?") for r in results})

    # 鈹€鈹€ Data extraction 鈹€鈹€
    # Figure 5(b): hop vs arrival_rate  (tag="fig5b")
    hop_by_ar = _aggregate(results, strategies, "arrival_rate", "avg_comm_delay", "fig5b")
    # Figure 5(c): hop vs lifetime  (tag="fig5c")
    hop_by_lt = _aggregate(results, strategies, "mean_lifetime", "avg_comm_delay", "fig5c")
    # Figure 6(b): acc vs arrival_rate  (tag="fig6b")
    acc_by_ar = _aggregate(results, strategies, "arrival_rate", "acceptance_rate", "fig6b")
    # Figure 6(c): acc vs lifetime  (tag="fig6c")
    acc_by_lt = _aggregate(results, strategies, "mean_lifetime", "acceptance_rate", "fig6c")

    # Time-series from stability runs (tag="stability")
    ts_acceptance: Dict[str, Tuple] = {}
    ts_hops: Dict[str, Tuple] = {}
    for r in results:
        if r.get("sweep_tag") != "stability":
            continue
        s = r.get("strategy", "")
        if "history" in r and len(r["history"]) > 0:
            h = r["history"]
            ts_acceptance[s] = ([p["time_step"] for p in h],
                                 [p["acceptance_rate"] for p in h])
        if "hop_series" in r and len(r["hop_series"]) > 0:
            hs = r["hop_series"]
            ts_hops[s] = ([p["time_step"] for p in hs],
                            [p["hops"] for p in hs])

    # ------------------------------------------------------------------
    #  Figure 5: Communication Hops (line plots)
    # ------------------------------------------------------------------
    fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(18, 5.2))

    try:
        _plot_lines(ax5a, ts_hops, strategies)
    except Exception:
        pass
    ax5a.set_xlabel("Time Step", fontsize=11)
    ax5a.set_ylabel("Hops", fontsize=11)
    ax5a.set_title("(a) Stability", fontsize=13, fontweight="bold")
    ax5a.set_ylim(0, 25)
    ax5a.grid(True, alpha=0.25)
    if ts_hops:
        ax5a.legend(fontsize=9)

    _plot_lines(ax5b, hop_by_ar, strategies, "s-", ms=6)
    ax5b.set_xlabel("Arrival Rate", fontsize=11)
    ax5b.set_ylabel("Avg Hops", fontsize=11)
    ax5b.set_title("(b) Traffic Surge", fontsize=13, fontweight="bold")
    ax5b.grid(True, alpha=0.25)
    if hop_by_ar: ax5b.legend(fontsize=9)

    _plot_lines(ax5c, hop_by_lt, strategies, "s-", ms=6)
    ax5c.set_xlabel("Mean Task Lifetime", fontsize=11)
    ax5c.set_ylabel("Avg Hops", fontsize=11)
    ax5c.set_title("(c) Task Lifetime", fontsize=13, fontweight="bold")
    ax5c.grid(True, alpha=0.25)
    if hop_by_lt: ax5c.legend(fontsize=9)

    fig5.suptitle("Figure 5: Communication Delay (Hops)", fontsize=15,
                   fontweight="bold", y=1.02)
    fig5.tight_layout()
    p = os.path.join(output_dir, "figure5_comm_hops.png")
    fig5.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig5)

    # ------------------------------------------------------------------
    #  Figure 6: Acceptance Rate  (a:line, b/c:bar)
    # ------------------------------------------------------------------
    fig6, (ax6a, ax6b, ax6c) = plt.subplots(1, 3, figsize=(18, 5.2))

    try:
        _plot_lines(ax6a, ts_acceptance, strategies)
    except Exception:
        pass
    ax6a.set_xlabel("Time Step", fontsize=11)
    ax6a.set_ylabel("Acceptance Rate", fontsize=11)
    ax6a.set_title("(a) Long-term Performance", fontsize=13, fontweight="bold")
    ax6a.set_ylim(0.4, 1.05)
    ax6a.grid(True, alpha=0.25)
    if ts_acceptance:
        ax6a.legend(fontsize=9)

    _plot_bars(ax6b, acc_by_ar, strategies)
    ax6b.set_xlabel("Arrival Rate", fontsize=11)
    ax6b.set_ylabel("Acceptance Rate", fontsize=11)
    ax6b.set_title("(b) Traffic Surge", fontsize=13, fontweight="bold")
    ax6b.set_ylim(0.4, 1.05)
    ax6b.grid(True, alpha=0.2, axis="y")
    if acc_by_ar: ax6b.legend(fontsize=9)

    _plot_bars(ax6c, acc_by_lt, strategies)
    ax6c.set_xlabel("Mean Task Lifetime", fontsize=11)
    ax6c.set_ylabel("Acceptance Rate", fontsize=11)
    ax6c.set_title("(c) Task Lifetime", fontsize=13, fontweight="bold")
    ax6c.set_ylim(0.4, 1.05)
    ax6c.grid(True, alpha=0.2, axis="y")
    if acc_by_lt: ax6c.legend(fontsize=9)

    fig6.suptitle("Figure 6: Service Acceptance Rate", fontsize=15,
                   fontweight="bold", y=1.02)
    fig6.tight_layout()
    p = os.path.join(output_dir, "figure6_acceptance_rate.png")
    fig6.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved {p}")
    plt.close(fig6)

    # 鈹€鈹€ JSON 鈹€鈹€
    p = os.path.join(output_dir, "evaluation_results.json")
    def _json_default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"  Saved {p}")


# =========================================================================
#  CLI
# =========================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="AgentVNE Evaluation Suite")
    parser.add_argument("--sn_topology",
                        default=os.path.join(script_dir, "topo", "SN_topology.json"))
    parser.add_argument("--sn_noderank",
                        default=os.path.join(script_dir, "topo", "SN_topology_noderank.json"))
    parser.add_argument("--vn_noderank",
                        default=os.path.join(script_dir, "Workflow_topo", "workflow1_noderank.json"))
    parser.add_argument("--workflow", action="append",
                        default=["workflow1=" + os.path.join(script_dir, "Workflow_topo", "workflow1_topo.json")])
    parser.add_argument("--strategies", nargs="+", default=["ga", "pretrain", "greedy", "agentvne"],
                        help="Strategy names: greedy, gal-vne, ga, grc, pretrain, agentvne")
    parser.add_argument("--pretrain_model",
                        default=os.path.join(script_dir, "pretrain_outputs", "checkpoint_latest.pt"))
    parser.add_argument("--finetuned_model",
                        default=os.path.join(script_dir, "finetuning_output", "policy_network_latest.pth"))
    parser.add_argument("--parameter", action="append", default=[])
    parser.add_argument("--arrival_rates", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--mean_lifetimes", nargs="+", type=float,
                        default=[20, 40, 60])
    parser.add_argument("--max_time_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir",
                        default=os.path.join(script_dir, "eval_output"))
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sweep_mode", type=str, default="grid",
                        choices=["grid", "arrival", "lifetime", "stability"])
    parser.add_argument("--run_all_paper", action="store_true", default=False,
                        help="Run 4 parameter sweeps + 1 stability run per paper specs")
    args = parser.parse_args()

    # parse workflow
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

    # read SN node count
    with open(args.sn_topology, "r", encoding="utf-8") as _f:
        _sn = json.load(_f)
    num_sn_nodes = len(_sn.get("nodes", []))

    extra = {
        "device": args.device,
        "model_path": args.pretrain_model,
        "finetuned_model_path": args.finetuned_model,
        "sn_noderank_path": args.sn_noderank,
        "vn_noderank_path": args.vn_noderank,
        "pop_size": 50, "generations": 100,
        "num_sn_nodes": num_sn_nodes,
    }

    strategies: Dict[str, Callable] = {}
    for sname in args.strategies:
        try:
            strategies[sname] = resolve_strategy(sname, extra)
        except Exception as e:
            _warn(f"{sname}: {e}")

    if not strategies:
        print("No valid strategies. Exiting."); return

    sweep_kwargs = {
        "max_time_steps": args.max_time_steps,
        "max_arrived_tasks": param_overrides.get("max_arrived_tasks", 20),
        "seed": args.seed,
        "verbose": True,
    }

    if args.run_all_paper:
        print("=== Paper Figure 5 & 6 Full Sweep ===\n")
        all_results: List[Dict] = []

        # ---- Fig5(b): arrival_rate sweep [0.05..0.50], lifetime=40 ----
        print("\n--- [fig5b] Arrival Rate -> Hops (lifetime=40) ---")
        all_results += parameter_sweep(strategies, args.sn_topology,
            workflow_types, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            mean_lifetimes, sweep_mode="arrival", fixed_lifetime=40.0,
            sweep_tag="fig5b", **sweep_kwargs)

        # ---- Fig5(c): lifetime sweep [15..40], arrival=0.1 ----
        print("\n--- [fig5c] Lifetime -> Hops (arrival=0.1) ---")
        all_results += parameter_sweep(strategies, args.sn_topology,
            workflow_types, arrival_rates,
            [15, 20, 25, 30, 35, 40], sweep_mode="lifetime",
            fixed_arrival=0.1, sweep_tag="fig5c", **sweep_kwargs)

        # ---- Fig6(b): arrival_rate bar [0.20,0.30,0.40], lifetime=40 ----
        print("\n--- [fig6b] Arrival Rate -> Acc (lifetime=40) ---")
        all_results += parameter_sweep(strategies, args.sn_topology,
            workflow_types, [0.2, 0.3, 0.4],
            mean_lifetimes, sweep_mode="arrival", fixed_lifetime=40.0,
            sweep_tag="fig6b", **sweep_kwargs)

        # ---- Fig6(c): lifetime bar [20,30,40], arrival=0.1 ----
        print("\n--- [fig6c] Lifetime -> Acc (arrival=0.1) ---")
        all_results += parameter_sweep(strategies, args.sn_topology,
            workflow_types, arrival_rates,
            [20, 30, 40], sweep_mode="lifetime",
            fixed_arrival=0.1, sweep_tag="fig6c", **sweep_kwargs)

        # ---- Stability run (10000 steps) ----
        print("\n--- [stability] Time-series (arrival=0.1, lifetime=40, steps=10000) ---")
        stab_kw = dict(sweep_kwargs)
        stab_kw["max_time_steps"] = 10000
        all_results += parameter_sweep(strategies, args.sn_topology,
            workflow_types, arrival_rates, mean_lifetimes,
            sweep_mode="stability", fixed_arrival=0.1, fixed_lifetime=40.0,
            sweep_tag="stability", **stab_kw)

        if args.plot:
            plot_results(all_results, output_dir=args.output_dir)
        print("\nDone.")
        return

    # Single sweep mode
    print("=" * 60)
    print("AgentVNE Evaluation Framework")
    print(f"  SN: {args.sn_topology}")
    print(f"  Workflows: {list(workflow_types.keys())}")
    print(f"  Strategies: {sorted(strategies.keys())}")
    print(f"  Arrival rates: {arrival_rates}")
    print(f"  Lifetimes: {mean_lifetimes}")
    print(f"  Max steps: {args.max_time_steps}")
    print(f"  Sweep mode: {args.sweep_mode}")
    print("=" * 60)

    results = parameter_sweep(strategies, args.sn_topology,
        workflow_types, arrival_rates, mean_lifetimes,
        sweep_mode=args.sweep_mode, **sweep_kwargs)
    if args.plot:
        plot_results(results, output_dir=args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=== Tester Quick Self-Test ===")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_args = argparse.Namespace(
            sn_topology=os.path.join(script_dir, "topo", "SN_topology.json"),
            sn_noderank=os.path.join(script_dir, "topo", "SN_topology_noderank.json"),
            vn_noderank=os.path.join(script_dir, "Workflow_topo", "workflow1_noderank.json"),
            workflow=["workflow1=" + os.path.join(script_dir, "Workflow_topo", "workflow1_topo.json")],
            strategies=["greedy"],
            pretrain_model=os.path.join(script_dir, "pretrain_outputs", "checkpoint_latest.pt"),
            finetuned_model=os.path.join(script_dir, "finetuning_output", "policy_network_latest.pth"),
            parameter=[], arrival_rates=[0.1], mean_lifetimes=[40.0],
            max_time_steps=300, seed=42,
            output_dir=os.path.join(script_dir, "eval_output"),
            plot=True, device="cpu", sweep_mode="arrival",
            run_all_paper=False,
        )
        wf_types: Dict[str, str] = {}
        for wf_spec in main_args.workflow:
            name, path = wf_spec.split("=", 1)
            if not os.path.isabs(path):
                path = os.path.join(script_dir, path)
            wf_types[name] = path
        with open(main_args.sn_topology, "r", encoding="utf-8") as _f:
            _sn = json.load(_f)
        extra = {
            "device": "cpu",
            "model_path": main_args.pretrain_model,
            "finetuned_model_path": main_args.finetuned_model,
            "sn_noderank_path": main_args.sn_noderank,
            "vn_noderank_path": main_args.vn_noderank,
            "pop_size": 50, "generations": 100,
            "num_sn_nodes": len(_sn.get("nodes", [])),
        }
        strats = {}
        for sn in main_args.strategies:
            try:
                strats[sn] = resolve_strategy(sn, extra)
            except Exception:
                pass
        results = parameter_sweep(strats, main_args.sn_topology,
            wf_types, main_args.arrival_rates, main_args.mean_lifetimes,
            max_time_steps=main_args.max_time_steps, max_arrived_tasks=20,
            seed=main_args.seed, verbose=True, sweep_mode="arrival")
        if main_args.plot:
            plot_results(results, output_dir=main_args.output_dir)
        print("\n=== Self-test complete ===")
    else:
        main()
