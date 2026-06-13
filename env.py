#!/usr/bin/env python3
"""
SimuVNE RL Environment - Phase 2 Implementation

Implements:
  - WorkflowGenerator: Poisson arrival + exponential lifetime VN generation
  - SimuVNEEnv: time-driven RL environment for virtual network embedding

Tightly follows the paper Section 3 (System Model) and Algorithm 1.
"""
import json
import os
import copy
import random
from collections import deque
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


# --- helpers -----------------------------------------------------------

def _load_json(path: str) -> Dict:
    """Load a JSON file with error handling."""
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(script_dir, path)
    else:
        abs_path = path
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    with open(abs_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_edge_index(links: List[Dict], directed: bool) -> torch.Tensor:
    """Build PyG edge_index [2, E] from link list."""
    edges: List[Tuple[int, int]] = []
    for link in links:
        u = int(link['source'])
        v = int(link['target'])
        edges.append((u, v))
        if not directed:
            edges.append((v, u))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _nodes_to_features(
    nodes: List[Dict],
    sn_max_capacity: Dict[str, float],
) -> torch.Tensor:
    """Map nodes to 6-dim normalised feature tensor [N, 6].

    Feature order: [cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]
    """
    feats: List[List[float]] = []
    max_cpu = sn_max_capacity.get('cpu_max', 1.0)
    max_mem = sn_max_capacity.get('mem_max', 1.0)
    max_disk = sn_max_capacity.get('disk_max', 1.0)
    max_bw = sn_max_capacity.get('bw_max', 1.0)
    max_comm = sn_max_capacity.get('comm_bw_max', 1.0)

    for n in nodes:
        cpu = float(n.get('cpu', 0.0)) / max_cpu
        mem = float(n.get('memory', 0.0)) / max_mem
        disk = float(n.get('disk', 0.0)) / max_disk
        bw = float(n.get('bandwidth', 1.0)) / max_bw
        comm = float(n.get('comm_bandwidth', bw * max_bw)) / max_comm
        feats.append([cpu, mem, disk, bw, comm, 0.0])

    return torch.tensor(feats, dtype=torch.float)


def _topology_to_pyg(topo: Dict, sn_max_capacity: Dict[str, float]) -> Data:
    """Convert topology JSON to PyG Data (normalised)."""
    nodes = topo['nodes']
    links = topo.get('links', [])
    directed = bool(topo.get('directed', False))
    x = _nodes_to_features(nodes, sn_max_capacity)
    edge_index = _build_edge_index(links, directed)
    return Data(x=x, edge_index=edge_index)


def _build_sn_id_to_idx(nodes: List[Dict]) -> Dict[int, int]:
    """Map SN node id -> list index."""
    mapping: Dict[int, int] = {}
    for idx, node in enumerate(nodes):
        sn_id = int(node.get('id', idx))
        mapping[sn_id] = idx
    return mapping


def _compute_sn_max_capacity(nodes: List[Dict]) -> Dict[str, float]:
    """Compute per-resource maxima from SN nodes."""
    caps = {'cpu_max': 1.0, 'mem_max': 1.0, 'disk_max': 1.0,
            'bw_max': 1.0, 'comm_bw_max': 1.0}
    for n in nodes:
        caps['cpu_max'] = max(caps['cpu_max'], float(n.get('cpu', 0.0)))
        caps['mem_max'] = max(caps['mem_max'], float(n.get('memory', 0.0)))
        caps['disk_max'] = max(caps['disk_max'], float(n.get('disk', 0.0)))
        caps['bw_max'] = max(caps['bw_max'], float(n.get('bandwidth', 0.0)))
        caps['comm_bw_max'] = max(caps['comm_bw_max'],
                                  float(n.get('comm_bandwidth',
                                              n.get('bandwidth', 0.0))))
    return caps


def _build_sn_graph(sn_nodes: List[Dict], sn_links: List[Dict]) -> nx.Graph:
    """Build undirected NetworkX graph for shortest-path lookups."""
    G = nx.Graph()
    for node in sn_nodes:
        G.add_node(int(node.get('id', len(G.nodes))),
                   cpu=float(node.get('cpu', 0.0)),
                   memory=float(node.get('memory', 0.0)),
                   disk=float(node.get('disk', 0.0)),
                   bandwidth=float(node.get('bandwidth', 1.0)),
                   comm_bandwidth=float(node.get('comm_bandwidth', 1.0)))
    for link in sn_links:
        u = int(link['source'])
        v = int(link['target'])
        w = float(link.get('weight', 1.0))
        bw = float(link.get('bandwidth', 1.0))
        G.add_edge(u, v, weight=w, bandwidth=bw)
    return G


# --- WorkflowGenerator --------------------------------------------------

class WorkflowGenerator:
    """Poisson-arrival + exponential-lifetime VN request generator.

    At each time step, samples the number of newly arriving VNs from
    Poisson(arrival_rate) and assigns a random type and lifetime.
    """

    def __init__(
        self,
        arrival_rate: float = 0.1,
        mean_lifetime: float = 50.0,
        workflow_types: Optional[Dict[str, str]] = None,
        max_arrived_tasks: int = 20,
        seed: Optional[int] = None,
    ):
        self.arrival_rate = arrival_rate
        self.mean_lifetime = mean_lifetime
        self.max_arrived_tasks = max_arrived_tasks
        self.rng = np.random.default_rng(seed)

        self.workflow_types: Dict[str, str] = workflow_types or {}
        self._workflow_caches: Dict[str, Dict] = {}

        self.active_vns: Dict[int, Dict] = {}
        self._next_vn_id = 0

    def add_workflow_type(self, name: str, topo_path: str) -> None:
        """Register a new workflow type (topology template)."""
        self.workflow_types[name] = topo_path
        if name in self._workflow_caches:
            del self._workflow_caches[name]

    def _load_workflow(self, name: str) -> Dict:
        """Load (cached) workflow topology dict."""
        if name not in self._workflow_caches:
            path = self.workflow_types[name]
            self._workflow_caches[name] = _load_json(path)
        return copy.deepcopy(self._workflow_caches[name])

    def generate_arrival_events(
        self, time_step: int
    ) -> List[Dict[str, Any]]:
        """Produce list of arrival events for *time_step*.

        Each event dict:
          - vn_id: int           unique VN identifier
          - type: str            workflow type name
          - topo: dict           VN topology (deep-copied)
          - lifetime: float      remaining time steps
          - constraint_node: int | None
        """
        n_arrivals = min(
            self.rng.poisson(self.arrival_rate),
            self.max_arrived_tasks,
        )
        events: List[Dict[str, Any]] = []
        available_types = list(self.workflow_types.keys())
        if not available_types:
            return events

        for _ in range(n_arrivals):
            vn_id = self._next_vn_id
            self._next_vn_id += 1
            wf_type = self.rng.choice(available_types)
            topo = self._load_workflow(wf_type)
            lifetime = self.rng.exponential(self.mean_lifetime)
            lifetime = int(max(1, round(lifetime)))

            constraint = None
            for node in topo.get('nodes', []):
                cn = node.get('constraint_node')
                if cn is not None:
                    constraint = int(cn)
                    break

            event = {
                'vn_id': vn_id,
                'type': wf_type,
                'topo': topo,
                'lifetime': lifetime,
                'constraint_node': constraint,
            }
            events.append(event)
            self.active_vns[vn_id] = {
                'type': wf_type,
                'remaining_lifetime': lifetime,
                'topo': copy.deepcopy(topo),
                'constraint_node': constraint,
                'placed': False,
                'mapping': {},
            }

        return events

    def step_time(self) -> Set[int]:
        """Advance one time step; return IDs of expired VNs."""
        expired: Set[int] = set()
        for vn_id in list(self.active_vns.keys()):
            self.active_vns[vn_id]['remaining_lifetime'] -= 1
            if self.active_vns[vn_id]['remaining_lifetime'] <= 0:
                expired.add(vn_id)
        for vn_id in expired:
            del self.active_vns[vn_id]
        return expired

    def get_active_count(self) -> int:
        """Number of currently active VNs."""
        return len(self.active_vns)

    def get_active_vns(self) -> List[int]:
        """Return list of active VN ids."""
        return list(self.active_vns.keys())

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset generator state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.active_vns.clear()
        self._next_vn_id = 0


# --- SimuVNEEnv ---------------------------------------------------------

class SimuVNEEnv:
    """Time-driven VNE RL environment.

    State space  - (sn_data, vn_data): two PyG Data objects (normalised).
    Action space - probability matrix [N_vn, N_sn] (model output).
    Reward       - acceptance bonus - comm-delay penalty - rejection cost.
    """

    def __init__(
        self,
        sn_topology_path: str,
        workflow_generator: WorkflowGenerator,
        seed: Optional[int] = None,
        max_time_steps: int = 1000,
        acceptance_reward: float = 10.0,
        rejection_penalty: float = -3.0,
        comm_delay_weight: float = 0.1,
    ):
        self.sn_topo_path = sn_topology_path
        self.wf_gen = workflow_generator
        self.max_time_steps = max_time_steps
        self.acceptance_reward = acceptance_reward
        self.rejection_penalty = rejection_penalty
        self.comm_delay_weight = comm_delay_weight
        self.rng = np.random.default_rng(seed)

        self._sn_topo_raw = _load_json(sn_topology_path)
        self._sn_nodes_original = copy.deepcopy(self._sn_topo_raw['nodes'])
        self._sn_links = self._sn_topo_raw.get('links', [])
        self._sn_max_cap = _compute_sn_max_capacity(self._sn_nodes_original)
        self.num_sn_nodes = len(self._sn_nodes_original)
        self._sn_id_to_idx = _build_sn_id_to_idx(self._sn_nodes_original)

        self.sn_graph = _build_sn_graph(self._sn_nodes_original, self._sn_links)

        self._sn_nodes: List[Dict] = []
        self._time_step: int = 0
        self._done: bool = False

        self.total_accepted: int = 0
        self.total_arrived: int = 0
        self.cumulative_reward: float = 0.0
        self.current_vn: Optional[Dict] = None
        self.current_vn_data: Optional[Data] = None
        self._pending_vns: deque = deque()

        self._placed_vns_registry: Dict[int, Dict] = {}

    @property
    def sn_nodes(self) -> List[Dict]:
        return self._sn_nodes

    @property
    def time_step(self) -> int:
        return self._time_step

    @property
    def done(self) -> bool:
        return self._done

    # --- state ----------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[Data, Optional[Data]]:
        """Reset environment to initial state.

        Returns (sn_data, vn_data) for the first arriving VN.
        If no VN arrives immediately, returns (sn_data, None).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.wf_gen.reset(seed=seed)

        self._sn_nodes = copy.deepcopy(self._sn_nodes_original)
        self._time_step = 0
        self._done = False
        self.total_accepted = 0
        self.total_arrived = 0
        self.cumulative_reward = 0.0
        self._pending_vns.clear()
        self._placed_vns_registry.clear()
        self.current_vn = None
        self.current_vn_data = None

        events = self.wf_gen.generate_arrival_events(0)
        for evt in events:
            self._pending_vns.append(evt)

        if self._pending_vns:
            self.current_vn = self._pending_vns.popleft()
            self.current_vn_data = self._vn_to_data(self.current_vn['topo'])
        else:
            self.current_vn = None
            self.current_vn_data = None

        return self.get_state()

    def get_state(self) -> Tuple[Data, Optional[Data]]:
        """Return current (sn_data, vn_data)."""
        sn_data = self._sn_to_data()
        vn_data = self.current_vn_data
        return sn_data, vn_data

    def _sn_to_data(self) -> Data:
        """Build PyG Data from current SN resource levels."""
        return _topology_to_pyg(
            {'nodes': self._sn_nodes, 'links': self._sn_links,
             'directed': self._sn_topo_raw.get('directed', False)},
            self._sn_max_cap,
        )

    def _vn_to_data(self, vn_topo: Dict) -> Data:
        """Build PyG Data for a VN topology."""
        return _topology_to_pyg(vn_topo, self._sn_max_cap)

    # --- resource checks ------------------------------------------

    def is_valid_assignment(
        self, vn_node_idx: int, sn_node_idx: int,
        temp_mapping: Optional[Dict[int, int]] = None,
    ) -> bool:
        """Check if SN node has enough resources for VN node."""
        if self.current_vn is None:
            return False
        vn_nodes = self.current_vn['topo']['nodes']
        vn_node = vn_nodes[vn_node_idx]
        sn_node = self._sn_nodes[sn_node_idx]

        demand_cpu = float(vn_node.get('cpu', 0.0))
        demand_mem = float(vn_node.get('memory', 0.0))
        demand_disk = float(vn_node.get('disk', 0.0))

        avail_cpu = float(sn_node.get('cpu', 0.0))
        avail_mem = float(sn_node.get('memory', 0.0))
        avail_disk = float(sn_node.get('disk', 0.0))

        if temp_mapping:
            for vm_idx, sm_idx in temp_mapping.items():
                if sm_idx == sn_node_idx:
                    vm_node = vn_nodes[vm_idx]
                    avail_cpu -= float(vm_node.get('cpu', 0.0))
                    avail_mem -= float(vm_node.get('memory', 0.0))
                    avail_disk -= float(vm_node.get('disk', 0.0))

        return (demand_cpu <= avail_cpu + 1e-9 and
                demand_mem <= avail_mem + 1e-9 and
                demand_disk <= avail_disk + 1e-9)

    def get_legal_mask(self, vn_node_idx: int) -> torch.Tensor:
        """Return binary mask [N_sn] indicating legal SN nodes for VN node.

        If the VN node has a constraint_node field, only the corresponding
        SN node (matched by ID) is legal.
        """
        if self.current_vn is None:
            return torch.zeros(self.num_sn_nodes, dtype=torch.bool)

        vn_node = self.current_vn['topo']['nodes'][vn_node_idx]
        constraint_node_id = vn_node.get('constraint_node')
        mask = torch.zeros(self.num_sn_nodes, dtype=torch.bool)

        if constraint_node_id is not None:
            constraint_id = int(constraint_node_id)
            if constraint_id in self._sn_id_to_idx:
                sn_idx = self._sn_id_to_idx[constraint_id]
                if self.is_valid_assignment(vn_node_idx, sn_idx):
                    mask[sn_idx] = True
            return mask

        for sn_idx in range(self.num_sn_nodes):
            if self.is_valid_assignment(vn_node_idx, sn_idx):
                mask[sn_idx] = True
        return mask

    # --- resource ops --------------------------------------------

    def _deduct_single(self, vn_node_idx: int, sn_node_idx: int) -> None:
        """Deduct resources for one VN to SN mapping."""
        vn_node = self.current_vn['topo']['nodes'][vn_node_idx]
        sn_node = self._sn_nodes[sn_node_idx]
        sn_node['cpu'] = float(sn_node.get('cpu', 0.0)) - float(vn_node.get('cpu', 0.0))
        sn_node['memory'] = float(sn_node.get('memory', 0.0)) - float(vn_node.get('memory', 0.0))
        sn_node['disk'] = float(sn_node.get('disk', 0.0)) - float(vn_node.get('disk', 0.0))

    def allocate_resources(self, mapping: Dict[int, int]) -> None:
        """Commit a VN to SN mapping (deduct resources)."""
        if self.current_vn is None:
            return
        vn_nodes = self.current_vn['topo']['nodes']
        for vn_idx, sn_idx in mapping.items():
            if 0 <= vn_idx < len(vn_nodes) and 0 <= sn_idx < len(self._sn_nodes):
                self._deduct_single(vn_idx, sn_idx)
        self.total_accepted += 1

    def release_resources(self, vn_topo: Dict, mapping: Dict[int, int]) -> None:
        """Release resources when a VN departs."""
        vn_nodes = vn_topo['nodes']
        for vn_idx, sn_idx in mapping.items():
            if 0 <= vn_idx < len(vn_nodes) and 0 <= sn_idx < len(self._sn_nodes):
                vn_node = vn_nodes[vn_idx]
                sn_node = self._sn_nodes[sn_idx]
                sn_node['cpu'] = float(sn_node.get('cpu', 0.0)) + float(vn_node.get('cpu', 0.0))
                sn_node['memory'] = float(sn_node.get('memory', 0.0)) + float(vn_node.get('memory', 0.0))
                sn_node['disk'] = float(sn_node.get('disk', 0.0)) + float(vn_node.get('disk', 0.0))

    # --- reward --------------------------------------------------

    def compute_reward(self, mapping: Dict[int, int], success: bool) -> float:
        """Compute scalar reward for a placement attempt."""
        if not success:
            return self.rejection_penalty

        reward = self.acceptance_reward
        comm_penalty = self._compute_comm_penalty(mapping)
        reward -= self.comm_delay_weight * comm_penalty
        return reward

    def _compute_comm_penalty(self, mapping: Dict[int, int]) -> float:
        """Compute communication delay penalty from mapped VN to SN edges."""
        if self.current_vn is None:
            return 0.0

        vn_links = self.current_vn['topo'].get('links', [])
        total_penalty = 0.0

        for link in vn_links:
            vn_src = int(link['source'])
            vn_dst = int(link['target'])
            if vn_src not in mapping or vn_dst not in mapping:
                continue
            sn_src = mapping[vn_src]
            sn_dst = mapping[vn_dst]
            try:
                path_len = nx.shortest_path_length(
                    self.sn_graph, source=sn_src, target=sn_dst,
                    weight='weight',
                )
                total_penalty += path_len
            except nx.NetworkXNoPath:
                total_penalty += float(self.num_sn_nodes)

        return total_penalty / max(len(vn_links), 1)

    # --- step ----------------------------------------------------

    def step(
        self, action: torch.Tensor,
    ) -> Tuple[Tuple[Data, Optional[Data]], float, bool, Dict[str, Any]]:
        """Execute one placement action.

        Args:
            action: probability matrix [N_vn, N_sn] from policy network.

        Returns:
            (state, reward, done, info)
        """
        info: Dict[str, Any] = {
            'accepted': False, 'mapping': {}, 'time_step': self._time_step
        }

        if self.current_vn is None:
            return self._advance_time()

        mapping = self._greedy_place(action)

        if mapping is None:
            reward = self.rejection_penalty
            info['accepted'] = False
        else:
            self.allocate_resources(mapping)
            reward = self.compute_reward(mapping, success=True)
            info['accepted'] = True
            info['mapping'] = mapping

            vn_id = self.current_vn['vn_id']
            if vn_id in self.wf_gen.active_vns:
                self.wf_gen.active_vns[vn_id]['placed'] = True
                self.wf_gen.active_vns[vn_id]['mapping'] = mapping.copy()
            self._placed_vns_registry[vn_id] = {
                'topo': copy.deepcopy(self.current_vn['topo']),
                'mapping': mapping.copy(),
            }

        self.cumulative_reward += reward
        self.total_arrived += 1

        (state, _, done, info2) = self._advance_time()
        info2.update(info)
        return state, reward, done, info2

    def _greedy_place(
        self, action: torch.Tensor,
    ) -> Optional[Dict[int, int]]:
        """Greedy VN to SN placement from action probability matrix.

        For each VN node, selects the highest-probability legal SN node.
        Constrained VN nodes are placed first (they have only one option).
        Returns None if any node cannot be placed.
        """
        if self.current_vn is None:
            return None

        vn_nodes = self.current_vn['topo']['nodes']
        N_vn = len(vn_nodes)
        mapping: Dict[int, int] = {}
        temp_mapping: Dict[int, int] = {}

        # order: constrained nodes first, then the rest
        constrained = [i for i in range(N_vn)
                       if vn_nodes[i].get('constraint_node') is not None]
        unconstrained = [i for i in range(N_vn) if i not in constrained]
        vn_order = constrained + unconstrained

        for vn_idx in vn_order:
            probs = action[vn_idx].detach().cpu().numpy()
            legal_mask = self.get_legal_mask(vn_idx)
            masked = probs * legal_mask.numpy().astype(np.float32)

            if masked.sum() < 1e-12:
                return None

            sn_candidates = list(range(self.num_sn_nodes))
            sn_candidates.sort(key=lambda j: masked[j], reverse=True)

            placed = False
            for sn_idx in sn_candidates:
                if not legal_mask[sn_idx]:
                    continue
                if self.is_valid_assignment(vn_idx, sn_idx,
                                            temp_mapping=temp_mapping):
                    mapping[vn_idx] = sn_idx
                    temp_mapping[vn_idx] = sn_idx
                    placed = True
                    break

            if not placed:
                return None

        return mapping

    def _advance_time(
        self,
    ) -> Tuple[Tuple[Data, Optional[Data]], float, bool, Dict[str, Any]]:
        """Advance to next pending VN or next time step."""
        info: Dict[str, Any] = {'accepted': False, 'mapping': {}, 'time_step': self._time_step}

        if self._pending_vns:
            self.current_vn = self._pending_vns.popleft()
            self.current_vn_data = self._vn_to_data(self.current_vn['topo'])
            return self.get_state(), 0.0, False, info

        self._time_step += 1

        if self._time_step >= self.max_time_steps:
            self._done = True
            self.current_vn = None
            self.current_vn_data = None
            return self.get_state(), 0.0, True, info

        expired = self.wf_gen.step_time()

        for vn_id, placed_info in list(self._placed_vns_registry.items()):
            if vn_id in expired:
                self.release_resources(
                    placed_info['topo'], placed_info['mapping'])
                del self._placed_vns_registry[vn_id]

        events = self.wf_gen.generate_arrival_events(self._time_step)
        for evt in events:
            self._pending_vns.append(evt)

        if self._pending_vns:
            self.current_vn = self._pending_vns.popleft()
            self.current_vn_data = self._vn_to_data(self.current_vn['topo'])
        else:
            self.current_vn = None
            self.current_vn_data = None

        return self.get_state(), 0.0, False, info

    # --- stats ---------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return environment statistics."""
        return {
            'time_step': self._time_step,
            'done': self._done,
            'total_accepted': self.total_accepted,
            'total_arrived': self.total_arrived,
            'acceptance_rate': (
                self.total_accepted / max(self.total_arrived, 1)
            ),
            'cumulative_reward': self.cumulative_reward,
            'active_vns': self.wf_gen.get_active_count(),
            'pending_vns': len(self._pending_vns),
        }


# --- quick test ---------------------------------------------------------

if __name__ == '__main__':
    print("=== SimuVNEEnv Quick Test ===")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sn_path = os.path.join(script_dir, 'topo', 'SN_topology.json')
    wf_path = os.path.join(script_dir, 'Workflow_topo', 'workflow1_topo.json')

    wf_gen = WorkflowGenerator(
        arrival_rate=0.2,
        mean_lifetime=30.0,
        workflow_types={'wf1': wf_path},
        max_arrived_tasks=5,
        seed=42,
    )

    env = SimuVNEEnv(
        sn_topology_path=sn_path,
        workflow_generator=wf_gen,
        seed=42,
        max_time_steps=50,
    )

    sn_data, vn_data = env.reset()
    print(f"SN nodes: {sn_data.x.shape[0]}, features: {sn_data.x.shape[1]}")
    if vn_data is not None:
        print(f"First VN nodes: {vn_data.x.shape[0]}")

    for t in range(20):
        if env.done:
            break
        if vn_data is not None:
            N_v = vn_data.x.shape[0]
            N_s = sn_data.x.shape[0]
            action = torch.rand(N_v, N_s)
            action = action / action.sum(dim=1, keepdim=True)
            (sn_data, vn_data), reward, done, info = env.step(action)
            print(f"  t={t:3d} | reward={reward:+.2f} | accepted={str(info['accepted']):5s} | "
                  f"rate={env.get_stats()['acceptance_rate']:.2%}")
        else:
            (sn_data, vn_data), reward, done, info = env.step(torch.zeros(1, 10))
            print(f"  t={t:3d} | no VN pending, advancing time")

    print(f"\nFinal stats: {env.get_stats()}")
    print("=== Test complete ===")
