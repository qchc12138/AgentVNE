import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import torch
import networkx as nx
from torch_geometric.data import Data


class WorkflowGenerator:
    """
    任务生成器：支持多种workflow类型，泊松到达，指数生存时间。
    """
    def __init__(self, 
                 workflow_types: Dict[str, str],
                 arrival_rate: float = 0.05,
                 mean_lifetime: float = 10.0,
                 seed: int = 42,
                 sn_capacity_for_norm: Dict[str, float] = None):
        """
        Args:
            workflow_types: {'workflow1': '/path/to/workflow1_topo.json', ...}
            arrival_rate: 泊松分布的λ参数（平均每单位时间到达任务数）
            mean_lifetime: 指数分布的均值（平均生存时间）
            seed: 随机种子
            sn_capacity_for_norm: SN最大容量字典，用于VN特征归一化
        """
        self.workflow_types = workflow_types
        self.arrival_rate = arrival_rate
        self.mean_lifetime = mean_lifetime
        self.rng = np.random.RandomState(seed)
        # SN容量用于归一化VN需求
        self.sn_capacity = sn_capacity_for_norm or {
            'cpu_max': 4.0,
            'mem_max': 4.0,
            'disk_max': 6.0,
            'bw_max': 10.0,
            'comm_bw_max': 10.0,
        }
        
    def check_arrival(self, time_unit: float = 1.0) -> bool:
        """按泊松分布决定当前时间单位是否有任务到达"""
        # 泊松分布：P(k events in time_unit) = (λt)^k * e^(-λt) / k!
        # 简化：生成泊松随机数，若>=1则有任务到达
        num_arrivals = self.rng.poisson(self.arrival_rate * time_unit)
        return num_arrivals > 0
    
    def sample_workflow_type(self) -> str:
        """随机选择一个workflow类型"""
        types = list(self.workflow_types.keys())
        return self.rng.choice(types)
    
    def sample_lifetime(self) -> float:
        """按指数分布采样任务生存时间"""
        return self.rng.exponential(self.mean_lifetime)
    
    def load_workflow_graph(self, workflow_type: str) -> Data:
        """加载指定类型的workflow图（特征已归一化）"""
        path = self.workflow_types[workflow_type]
        with open(path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        nodes = js['nodes']
        node_id_map = {int(n['id']): idx for idx, n in enumerate(nodes)}
        x_list: List[List[float]] = []
        constraint_nodes: List[Optional[int]] = []  # 存储每个VN节点的constraint_node信息
        
        # 使用SN容量归一化VN需求
        for n in nodes:
            x_list.append([
                float(n.get('cpu', 0.0)) / (self.sn_capacity['cpu_max'] + 1e-8),
                float(n.get('memory', 0.0)) / (self.sn_capacity['mem_max'] + 1e-8),
                float(n.get('disk', 0.0)) / (self.sn_capacity['disk_max'] + 1e-8),
                float(n.get('bandwidth', 1.0)) / (self.sn_capacity['bw_max'] + 1e-8),
                float(n.get('comm_bandwidth', 1.0)) / (self.sn_capacity['comm_bw_max'] + 1e-8),
                0.0,
            ])
            # 保存constraint_node信息（如果存在）
            constraint_node_id = n.get('constraint_node')
            if constraint_node_id is not None:
                constraint_nodes.append(int(constraint_node_id))
            else:
                constraint_nodes.append(None)
        x = torch.tensor(x_list, dtype=torch.float)
        edges = js['links']
        src, dst = [], []
        directed = js.get('directed', True)
        for e in edges:
            u = node_id_map[int(e['source'])]
            v = node_id_map[int(e['target'])]
            src.append(u); dst.append(v)
            if not directed:
                src.append(v); dst.append(u)
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        
        # 创建Data对象，并添加constraint_nodes属性
        data = Data(x=x, edge_index=edge_index)
        data.constraint_nodes = constraint_nodes  # 存储每个VN节点对应的constraint_node SN节点ID
        return data


class PlacementResult:
    """放置结果与环境反馈的简单容器。"""
    def __init__(self,
                 success: bool,
                 mapping: Dict[int, int],
                 vn_paths: List[Tuple[int, int, List[int]]],
                 reward: float,
                 info: Optional[Dict[str, Any]] = None):
        self.success = success
        self.mapping = mapping
        self.vn_paths = vn_paths
        self.reward = reward
        self.info = info or {}


class SimuVNEEnv:
    """
    强化学习环境（时间驱动版本）：
    - 维护底层SN图的剩余资源与链路带宽
    - 按时间单位推进，每单位时间：
      1) 检查并移除到期任务，恢复资源
      2) 检查是否有新任务到达（泊松）
      3) 若有任务到达，调用策略放置
      4) 计算 r_t（所有存活workflow的跳数均值）
    - 收集20个任务到达后结束
    """

    def __init__(self,
                 sn_topology_path: str,
                 device: str = 'cpu',
                 penalty: float = -10.0,
                 max_arrived_tasks: int = 20):
        self.device = torch.device(device)
        self.penalty = float(penalty)
        self.max_arrived_tasks = int(max_arrived_tasks)

        self.G_sn = self._load_sn(sn_topology_path)
        self._sn_initial_state = {}  # 保存初始资源用于reset
        self._sn_max_capacity = {}  # 保存SN最大容量用于归一化
        self._init_sn_residuals()

        self.current_time = 0.0
        self.arrived_count = 0  # 已到达任务数
        self.accepted_count = 0  # 已成功放置任务数
        # 存活任务列表：每项 {'vn': Data, 'mapping': dict, 'paths': list, 'expire_time': float, 'task_id': int}
        self.active_workflows: List[Dict[str, Any]] = []
        self.traj: List[Dict[str, Any]] = []  # 每个时间单位的轨迹：S_t, A_t, r_t, done, ...

    def _load_sn(self, path: str) -> nx.Graph:
        with open(path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        directed = js.get('directed', False)
        G = nx.DiGraph() if directed else nx.Graph()
        for n in js['nodes']:
            G.add_node(int(n['id']),
                       cpu=float(n.get('cpu', 0.0)),
                       memory=float(n.get('memory,', n.get('memory', 0.0))),
                       disk=float(n.get('disk', 0.0)),
                       bandwidth=float(n.get('bandwidth', 0.0)),
                       comm_bandwidth=float(n.get('comm_bandwidth', 0.0)))
        for e in js['links']:
            u = int(e['source']); v = int(e['target'])
            G.add_edge(u, v,
                       weight=float(e.get('weight', 1.0)),
                       bandwidth=float(e.get('bandwidth', 0.0)))
            if isinstance(G, nx.DiGraph):
                # 若为有向，确保对称链路可按需要扩展；此处按输入为准
                pass
        return G

    def _init_sn_residuals(self):
        """初始化并保存SN初始资源，同时计算最大容量用于归一化"""
        max_cpu = 0.0
        max_mem = 0.0
        max_disk = 0.0
        max_bw = 0.0
        max_comm_bw = 0.0
        
        for n in self.G_sn.nodes:
            data = self.G_sn.nodes[n]
            data['cpu_res'] = float(data.get('cpu', 0.0))
            data['mem_res'] = float(data.get('memory', 0.0))
            data['disk_res'] = float(data.get('disk', 0.0))
            self._sn_initial_state[n] = {
                'cpu': data['cpu_res'],
                'mem': data['mem_res'],
                'disk': data['disk_res'],
            }
            # 更新最大值
            max_cpu = max(max_cpu, data['cpu_res'])
            max_mem = max(max_mem, data['mem_res'])
            max_disk = max(max_disk, data['disk_res'])
            max_bw = max(max_bw, float(data.get('bandwidth', 0.0)))
            max_comm_bw = max(max_comm_bw, float(data.get('comm_bandwidth', 0.0)))
            
        for u, v in self.G_sn.edges:
            data = self.G_sn.edges[u, v]
            data['bw_res'] = float(data.get('bandwidth', 0.0))
            self._sn_initial_state[(u, v)] = {'bw': data['bw_res']}
            max_bw = max(max_bw, data['bw_res'])
        
        # 保存最大容量（用于归一化）
        self._sn_max_capacity = {
            'cpu_max': max_cpu if max_cpu > 0 else 1.0,
            'mem_max': max_mem if max_mem > 0 else 1.0,
            'disk_max': max_disk if max_disk > 0 else 1.0,
            'bw_max': max_bw if max_bw > 0 else 1.0,
            'comm_bw_max': max_comm_bw if max_comm_bw > 0 else 1.0,
        }

    def reset(self):
        """重置环境"""
        # 恢复SN资源
        for n in self.G_sn.nodes:
            init = self._sn_initial_state[n]
            self.G_sn.nodes[n]['cpu_res'] = init['cpu']
            self.G_sn.nodes[n]['mem_res'] = init['mem']
            self.G_sn.nodes[n]['disk_res'] = init['disk']
        for u, v in self.G_sn.edges:
            init = self._sn_initial_state[(u, v)]
            self.G_sn.edges[u, v]['bw_res'] = init['bw']
        
        self.current_time = 0.0
        self.arrived_count = 0
        self.accepted_count = 0
        self.active_workflows = []
        self.traj = []

    def _shortest_path(self, u: int, v: int) -> Optional[List[int]]:
        try:
            return nx.shortest_path(self.G_sn, source=u, target=v, weight='weight')
        except nx.NetworkXNoPath:
            return None

    def _check_node_feasible(self, vn: Data, mapping: Dict[int, int]) -> bool:
        """按 VN 节点需求从高到低的顺序检查资源可行性"""
        vn_demands = []
        for vn_node, sn_node in mapping.items():
            feats = vn.x[vn_node]
            norm_sum = float(feats[0].item() + feats[1].item() + feats[2].item())
            cpu = float(feats[0].item()) * (self._sn_max_capacity['cpu_max'] + 1e-8)
            mem = float(feats[1].item()) * (self._sn_max_capacity['mem_max'] + 1e-8)
            disk = float(feats[2].item()) * (self._sn_max_capacity['disk_max'] + 1e-8)
            vn_demands.append((norm_sum, vn_node, sn_node, cpu, mem, disk))
        
        vn_demands.sort(key=lambda x: x[0], reverse=True)
        temp_sn_res = {}
        for n in self.G_sn.nodes:
            nd = self.G_sn.nodes[n]
            temp_sn_res[n] = {
                'cpu': nd['cpu_res'],
                'mem': nd['mem_res'],
                'disk': nd['disk_res'],
            }
        
        # 按顺序检查并扣减资源
        for _, vn_node, sn_node, cpu, mem, disk in vn_demands:
            res = temp_sn_res[sn_node]
            if cpu > res['cpu'] + 1e-9: return False
            if mem > res['mem'] + 1e-9: return False
            if disk > res['disk'] + 1e-9: return False
            # 扣减资源
            res['cpu'] -= cpu
            res['mem'] -= mem
            res['disk'] -= disk
        
        return True

    def _compute_paths_and_bw_demand(self, vn: Data, mapping: Dict[int, int]) -> Optional[List[Tuple[int, int, List[int]]]]:
        # 对每条VN链路，找到映射的SN端点的最短路径
        edge_index = vn.edge_index
        vn_paths: List[Tuple[int, int, List[int]]] = []
        E = edge_index.size(1)
        for i in range(E):
            u = int(edge_index[0, i].item())
            v = int(edge_index[1, i].item())
            sn_u = mapping[u]
            sn_v = mapping[v]
            if sn_u == sn_v:
                vn_paths.append((sn_u, sn_v, [sn_u]))
                continue
            path = self._shortest_path(sn_u, sn_v)
            if path is None:
                return None
            vn_paths.append((sn_u, sn_v, path))
        return vn_paths

    def _check_link_feasible(self, vn: Data, vn_paths: List[Tuple[int, int, List[int]]]) -> bool:
        # 简化：每条VN链路带宽=1（或由x的某一维度给出）。这里取 x 的第 3 维作为带宽需求（若存在），否则 1。
        # 对路径上的每条SN边扣减需求，先只校验不扣减。
        bw_demands: List[float] = []
        for i, (su, sv, path) in enumerate(vn_paths):
            # 选择一个带宽需求来源：若 vn.x 有第四维，则取 abs 第四维，否则 1.0
            bw = 1.0
            if vn.x.size(1) >= 4:
                # 防止为0，至少取正
                bw = float(abs(vn.x[min(i, vn.x.size(0)-1), 3].item()))
                if bw <= 0:
                    bw = 1.0
            bw_demands.append(bw)

        # 按边累计需求
        edge_need: Dict[Tuple[int, int], float] = {}
        for (su, sv, path), bw in zip(vn_paths, bw_demands):
            if len(path) <= 1:
                continue
            for a, b in zip(path[:-1], path[1:]):
                key = (a, b) if (a, b) in self.G_sn.edges else (b, a)
                edge_need[key] = edge_need.get(key, 0.0) + bw

        for (a, b), need in edge_need.items():
            if need > self.G_sn.edges[a, b]['bw_res'] + 1e-9:
                return False
        return True

    def _apply_mapping(self, vn: Data, mapping: Dict[int, int], vn_paths: List[Tuple[int, int, List[int]]]):
        """
        按 VN 节点需求从高到低的顺序扣减资源
        """
        # 计算每个VN节点的绝对需求和优先级
        vn_demands = []
        for vn_node, sn_node in mapping.items():
            feats = vn.x[vn_node]
            norm_sum = float(feats[0].item() + feats[1].item() + feats[2].item())
            cpu = float(feats[0].item()) * (self._sn_max_capacity['cpu_max'] + 1e-8)
            mem = float(feats[1].item()) * (self._sn_max_capacity['mem_max'] + 1e-8)
            disk = float(feats[2].item()) * (self._sn_max_capacity['disk_max'] + 1e-8)
            vn_demands.append((norm_sum, vn_node, sn_node, cpu, mem, disk))
        
        # 按归一化需求从高到低排序
        vn_demands.sort(key=lambda x: x[0], reverse=True)
        
        # 在实际扣减前做一次严格校验，防止资源被超额扣减
        temp_sn_res_check = {
            n: {
                'cpu': self.G_sn.nodes[n]['cpu_res'],
                'mem': self.G_sn.nodes[n]['mem_res'],
                'disk': self.G_sn.nodes[n]['disk_res'],
            }
            for n in self.G_sn.nodes
        }

        for _, vn_node, sn_node, cpu, mem, disk in vn_demands:
            res = temp_sn_res_check[sn_node]
            if cpu > res['cpu'] + 1e-9 or mem > res['mem'] + 1e-9 or disk > res['disk'] + 1e-9:
                raise ValueError(
                    f"Resource over-allocation detected before deduction: "
                    f"SN {sn_node} available CPU={res['cpu']:.4f}, MEM={res['mem']:.4f}, DISK={res['disk']:.4f}; "
                    f"VN{vn_node} demands CPU={cpu:.4f}, MEM={mem:.4f}, DISK={disk:.4f}"
                )
            res['cpu'] -= cpu
            res['mem'] -= mem
            res['disk'] -= disk

        # 按顺序扣减资源
        for _, vn_node, sn_node, cpu, mem, disk in vn_demands:
            nd = self.G_sn.nodes[sn_node]
            nd['cpu_res'] -= cpu
            nd['mem_res'] -= mem
            nd['disk_res'] -= disk

        # 扣减链路带宽（已禁用 - 不追踪带宽资源）
        # bw_demands: List[float] = []
        # for i, (su, sv, path) in enumerate(vn_paths):
        #     bw = 1.0
        #     if vn.x.size(1) >= 4:
        #         bw = float(abs(vn.x[min(i, vn.x.size(0)-1), 3].item()))
        #         if bw <= 0:
        #             bw = 1.0
        #     bw_demands.append(bw)
        #
        # for (su, sv, path), bw in zip(vn_paths, bw_demands):
        #     if len(path) <= 1:
        #         continue
        #     for a, b in zip(path[:-1], path[1:]):
        #         key = (a, b) if (a, b) in self.G_sn.edges else (b, a)
        #         self.G_sn.edges[key[0], key[1]]['bw_res'] -= bw

    def _release_workflow(self, wf: Dict[str, Any]):
        """
        释放一个workflow占用的资源
        按 VN 节点需求从高到低的顺序恢复资源（与扣减顺序一致）
        """
        mapping = wf['mapping']
        vn: Data = wf['vn']
        vn_paths = wf['paths']
        
        # 计算每个VN节点的绝对需求和优先级
        vn_demands = []
        for vn_node, sn_node in mapping.items():
            feats = vn.x[vn_node]
            norm_sum = float(feats[0].item() + feats[1].item() + feats[2].item())
            cpu = float(feats[0].item()) * (self._sn_max_capacity['cpu_max'] + 1e-8)
            mem = float(feats[1].item()) * (self._sn_max_capacity['mem_max'] + 1e-8)
            disk = float(feats[2].item()) * (self._sn_max_capacity['disk_max'] + 1e-8)
            vn_demands.append((norm_sum, vn_node, sn_node, cpu, mem, disk))
        
        # 按归一化需求从高到低排序
        vn_demands.sort(key=lambda x: x[0], reverse=True)
        
        # 按顺序恢复资源
        for _, vn_node, sn_node, cpu, mem, disk in vn_demands:
            nd = self.G_sn.nodes[sn_node]
            nd['cpu_res'] += cpu
            nd['mem_res'] += mem
            nd['disk_res'] += disk
        
        # 恢复链路带宽（已禁用 - 不追踪带宽资源）
        # bw_demands: List[float] = []
        # for i, (su, sv, path) in enumerate(vn_paths):
        #     bw = 1.0
        #     if vn.x.size(1) >= 4:
        #         bw = float(abs(vn.x[min(i, vn.x.size(0)-1), 3].item()))
        #         if bw <= 0:
        #             bw = 1.0
        #     bw_demands.append(bw)
        # 
        # for (su, sv, path), bw in zip(vn_paths, bw_demands):
        #     if len(path) <= 1:
        #         continue
        #     for a, b in zip(path[:-1], path[1:]):
        #         key = (a, b) if (a, b) in self.G_sn.edges else (b, a)
        #         self.G_sn.edges[key[0], key[1]]['bw_res'] += bw
    
    def _check_and_remove_expired(self):
        """检查并移除到期任务"""
        remaining = []
        for wf in self.active_workflows:
            if wf['expire_time'] <= self.current_time:
                # 到期，释放资源
                self._release_workflow(wf)
            else:
                remaining.append(wf)
        self.active_workflows = remaining

    def get_sn_max_capacity(self) -> Dict[str, float]:
        """获取SN最大容量（用于VN特征归一化）"""
        return self._sn_max_capacity.copy()
    
    def _compute_rt(self) -> float:
        """
        跳数统计（考虑资源竞争）：
        1. 统计每条SN link被多少个workflow使用
        2. 对于每个workflow，其路径上的每条link如果被k个workflow共享，则算k跳
        3. r_t = -total_hops / 当前存活的workflow数量（负值，跳数越少，r_t 越大）
        """
        num_workflows = len(self.active_workflows)
        if num_workflows == 0:
            return 0.0
        
        # 第一步：统计每条SN边被多少个workflow使用
        # edge_usage_count: {(node_a, node_b): count}
        edge_usage_count: Dict[Tuple[int, int], int] = {}
        
        for wf in self.active_workflows:
            vn_paths = wf['paths']
            # 记录当前workflow使用了哪些SN边（避免重复计数）
            edges_in_this_wf: set = set()
            
            for su, sv, path in vn_paths:
                if len(path) <= 1:
                    continue
                # 遍历路径上的每条边
                for a, b in zip(path[:-1], path[1:]):
                    # 标准化边的表示（无向图）
                    edge = (min(a, b), max(a, b))
                    edges_in_this_wf.add(edge)
            
            # 统计这个workflow使用的所有边
            for edge in edges_in_this_wf:
                edge_usage_count[edge] = edge_usage_count.get(edge, 0) + 1
        
        # 第二步：计算每个workflow的加权跳数
        total_hops = 0.0
        
        for wf in self.active_workflows:
            vn_paths = wf['paths']
            workflow_hops = 0.0
            
            for su, sv, path in vn_paths:
                if len(path) <= 1:
                    # 两个VN节点映射到同一个SN节点，跳数为0
                    continue
                
                # 遍历路径上的每条边
                for a, b in zip(path[:-1], path[1:]):
                    # 标准化边的表示
                    edge = (min(a, b), max(a, b))
                    # 该边被k个workflow共享，则该边贡献k跳
                    k = edge_usage_count.get(edge, 1)
                    workflow_hops += k
            
            total_hops += workflow_hops
        
        # 第三步：按workflow数量平均，并取负值
        # r_t 为负数，跳数越少越好（越接近0）
        avg_hops = total_hops / float(num_workflows)
        r_t = -avg_hops
        return r_t
    
    def get_sn_state(self) -> Data:
        """获取当前SN状态的Data表示（节点特征包含归一化后的剩余资源）"""
        node_list = sorted(self.G_sn.nodes())
        x_list = []
        for n in node_list:
            nd = self.G_sn.nodes[n]
            # 获取初始容量用于归一化
            init = self._sn_initial_state[n]
            # 归一化：剩余资源 / 初始容量
            x_list.append([
                nd['cpu_res'] / (init['cpu'] + 1e-8),
                nd['mem_res'] / (init['mem'] + 1e-8),
                nd['disk_res'] / (init['disk'] + 1e-8),
                nd.get('bandwidth', 0.0) / (self._sn_max_capacity['bw_max'] + 1e-8),
                nd.get('comm_bandwidth', 0.0) / (self._sn_max_capacity['comm_bw_max'] + 1e-8),
                0.0,
            ])
        x = torch.tensor(x_list, dtype=torch.float)
        
        # 构建edge_index
        edges = list(self.G_sn.edges())
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            src, dst = [], []
            for u, v in edges:
                src.append(node_to_idx[u])
                dst.append(node_to_idx[v])
                if not self.G_sn.is_directed():
                    src.append(node_to_idx[v])
                    dst.append(node_to_idx[u])
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)

    def try_place_task(self,
                       vn: Data,
                       action_mapping: Dict[int, int],
                       lifetime: float,
                       task_id: int) -> Tuple[bool, float]:
        """
        尝试放置任务，返回(是否成功, r_t)
        """
        # 1) 节点资源校验
        if not self._check_node_feasible(vn, action_mapping):
            return False, self.penalty
        
        # 2) 计算VN链路对应的SN最短路径（不检查带宽限制）
        vn_paths = self._compute_paths_and_bw_demand(vn, action_mapping)
        if vn_paths is None:
            # 如果SN中不存在路径（网络不连通），则放置失败
            return False, self.penalty
        # 注意：已移除带宽资源检查，允许带宽超额使用
        
        # 3) 扣减资源，加入存活集合
        self._apply_mapping(vn, action_mapping, vn_paths)
        expire_time = self.current_time + lifetime
        self.active_workflows.append({
            'vn': vn,
            'mapping': action_mapping,
            'paths': vn_paths,
            'expire_time': expire_time,
            'task_id': task_id,
        })
        self.accepted_count += 1
        
        # 4) 计算 r_t（包含所有存活workflow）
        r_t = self._compute_rt()
        return True, r_t
    
    def step_time(self, time_delta: float = 1.0):
        """推进时间，返回当前是否已达到终止条件（20个任务到达）"""
        self.current_time += time_delta
        # 检查并移除到期任务
        self._check_and_remove_expired()
        return self.arrived_count >= self.max_arrived_tasks
    
    def is_done(self) -> bool:
        """检查是否已收集到足够的任务"""
        return self.arrived_count >= self.max_arrived_tasks

    def compute_final_return(self) -> float:
        """
        计算最终回报：
        final_reward = T_total / T_p * sum_{t=0..t_f} r_t
        注意：r_t 已经是负数（-avg_hops），所以不需要再加负号
        跳数越少，r_t越大（越接近0），final_reward越大（越接近0）
        """
        T_total = float(max(1, self.max_arrived_tasks))
        T_p = float(max(1, self.accepted_count))
        sum_rt = sum(float(x['r_t']) for x in self.traj)
        # 去掉负号，因为 r_t 已经是负数
        final_reward = T_total / T_p * sum_rt
        return final_reward


