# GAL (Greedy Allocation Algorithm) 算法实现详解

## 一、算法概述

**GAL (Greedy Allocation Algorithm)** 是一种基于贪心策略的虚拟网络嵌入（VNE）算法，使用 **SN NodeRank** 来指导节点放置决策。

**核心思想**：
1. 使用 SN NodeRank 评估 SN 节点的"重要性"和"资源丰富度"
2. 对于每个 VN 节点，选择 NodeRank 最高的可用 SN 节点
3. 采用两阶段放置策略：先放置非约束节点，再放置约束节点

---

## 二、算法架构

### 2.1 主要组件

1. **GreedyAllocator** (`baselines/GAL.py`)
   - 核心贪心放置算法
   - 使用 SN NodeRank 进行节点选择

2. **NodeRank 计算** (`baselines/noderank_utils.py`)
   - 计算 SN 节点的 NodeRank 值
   - 基于资源评估和邻居传播

3. **约束节点处理** (`tests/constraint_handler.py`)
   - 分离约束节点和非约束节点
   - 两阶段放置策略

4. **策略包装** (`tests/tester_gal.py`)
   - 将 GAL 算法包装为统一的策略接口

---

## 三、SN NodeRank 计算

### 3.1 算法步骤

**代码位置**：`baselines/noderank_utils.py` 第21-101行

#### 步骤1：计算初始资源评估 H(u)

```python
H(u) = cpu * comm_bandwidth
```

**说明**：
- `cpu`：SN 节点的 CPU 容量
- `comm_bandwidth`：SN 节点的通信带宽（优先使用 `comm_bandwidth`，如果没有则使用 `bandwidth`）
- 初始 NodeRank：`NR = H / sum(H)`（归一化）

#### 步骤2：构建无向邻接矩阵

```python
# 将 SN 图转换为无向图
# 构建邻接列表 adjacency[u] = [v1, v2, ...]
```

**说明**：
- 如果 SN 图是有向图，先转换为无向图
- 构建邻接列表，记录每个节点的邻居

#### 步骤3：计算前向概率 pF

```python
pF[u, v] = H(v) / sum(H(neighbors(u)))
```

**说明**：
- `pF[u, v]`：从节点 u 到节点 v 的前向概率
- 只在邻居节点之间分布
- 概率与邻居节点的 H 值成正比

#### 步骤4：传播和归一化

```python
PF_U = 0.20  # 传播因子
for _ in range(2):  # 传播两轮
    NR_next = NR_curr + PF_U * (pF @ NR_curr)
    NR_curr = NR_next / sum(NR_next)  # 归一化
```

**说明**：
- 进行两轮传播，每轮都归一化
- 传播因子 `PF_U = 0.20`，控制传播强度

#### 步骤5：三次幂强化

```python
NR_final = NR_curr ** 3
NR_final = NR_final / sum(NR_final)  # 最终归一化
```

**说明**：
- 做三次幂，强化 NodeRank 值的差异
- 最终归一化，确保和为1

### 3.2 NodeRank 的特点

1. **资源导向**：NodeRank 高的节点通常具有更高的 CPU 和带宽资源
2. **拓扑感知**：考虑节点的邻居关系，资源丰富的邻居会提升节点的 NodeRank
3. **归一化**：所有 NodeRank 值归一化，总和为1

---

## 四、GAL 贪心放置算法

### 4.1 算法流程

**代码位置**：`baselines/GAL.py` 第40-156行

#### 阶段1：分离约束节点和非约束节点

```python
from tests.constraint_handler import separate_constraint_nodes, place_constraint_nodes

non_constraint_indices, constraint_indices, constraint_mapping = separate_constraint_nodes(vn)
```

**说明**：
- 分离约束节点（必须映射到指定 SN 节点）和非约束节点（可以自由选择）
- `constraint_mapping`：约束节点的映射关系 `{vn_idx: sn_node_id}`

#### 阶段2：贪心放置非约束节点

```python
# 计算非约束节点的需求
vn_demands = []
for vn_idx in non_constraint_indices:
    feats = vn.x[vn_idx]
    abs_cpu = norm_cpu * (sn_max_capacity['cpu_max'] + 1e-8)
    abs_mem = norm_mem * (sn_max_capacity['mem_max'] + 1e-8)
    abs_disk = norm_disk * (sn_max_capacity['disk_max'] + 1e-8)
    vn_demands.append({
        'vn_node': vn_idx,
        'abs_cpu': abs_cpu,
        'abs_mem': abs_mem,
        'abs_disk': abs_disk,
    })
```

**说明**：
- 将归一化的 VN 节点需求转换为绝对需求
- 使用 SN 最大容量进行反归一化

#### 阶段3：为每个非约束节点选择 SN 节点

```python
for vn_info in vn_demands:
    vn_node = vn_info['vn_node']
    demand_cpu = vn_info['abs_cpu']
    demand_mem = vn_info['abs_mem']
    demand_disk = vn_info['abs_disk']
    
    # 获取所有满足资源需求的 SN 节点
    sn_nodes_with_rank = []
    for sn_node in self.env.G_sn.nodes:
        # 检查资源是否满足
        if (res_cpu >= demand_cpu - 1e-9 and 
            res_mem >= demand_mem - 1e-9 and 
            res_disk >= demand_disk - 1e-9):
            noderank = float(self.sn_noderank[sn_idx])
            sn_nodes_with_rank.append({
                'sn_node': sn_node,
                'noderank': noderank,
            })
    
    # 按 NodeRank 降序排序，选择最高的
    sn_nodes_with_rank.sort(key=lambda x: x['noderank'], reverse=True)
    best_sn = sn_nodes_with_rank[0]['sn_node']
    non_constraint_mapping[vn_node] = best_sn
    
    # 临时扣减资源（后续会恢复）
    nd = self.env.G_sn.nodes[best_sn]
    nd['cpu_res'] -= demand_cpu
    nd['mem_res'] -= demand_mem
    nd['disk_res'] -= demand_disk
    temporary_deductions.append((best_sn, demand_cpu, demand_mem, demand_disk))
```

**关键点**：
1. **资源检查**：只考虑满足资源需求的 SN 节点
2. **NodeRank 排序**：按 NodeRank 降序排序，选择最高的
3. **临时扣减**：记录临时资源扣减，如果后续失败可以回滚

#### 阶段4：恢复临时扣减

```python
_restore_temporary_deductions()
```

**说明**：
- 恢复临时扣减的资源
- 因为后续会统一通过 `env._apply_mapping` 应用映射

#### 阶段5：放置约束节点

```python
success, full_mapping, failure_reason = place_constraint_nodes(
    self.env, vn, non_constraint_mapping, constraint_mapping
)
```

**说明**：
- 在非约束节点映射的基础上，放置约束节点
- 检查约束节点是否可以放置到指定的 SN 节点
- 如果失败，返回失败原因

#### 阶段6：验证和应用映射

```python
# 验证映射并计算路径
vn_paths = self.env._compute_paths_and_bw_demand(vn, full_mapping)
if vn_paths is None:
    return False, {}, self.env.penalty

# 应用映射（扣减资源）
self.env._apply_mapping(vn, full_mapping, vn_paths)

# 返回成功
return True, full_mapping, 0.0  # r_t在外部计算
```

---

## 五、算法特点

### 5.1 贪心策略

**特点**：
- 每次选择 NodeRank 最高的可用 SN 节点
- 不考虑全局最优，只考虑局部最优
- 计算速度快，适合实时决策

**优点**：
- 时间复杂度低：O(VN_nodes × SN_nodes)
- 实现简单，易于理解
- 对于资源丰富的场景效果较好

**缺点**：
- 可能陷入局部最优
- 不考虑后续任务的影响

### 5.2 NodeRank 指导

**特点**：
- 使用 NodeRank 评估 SN 节点的"重要性"
- NodeRank 高的节点通常资源更丰富，拓扑位置更好
- 优先选择 NodeRank 高的节点

**优势**：
- 资源导向：优先使用资源丰富的节点
- 拓扑感知：考虑节点的邻居关系
- 自适应：NodeRank 会根据资源使用情况动态变化（如果资源被扣减）

### 5.3 两阶段放置

**特点**：
- 先放置非约束节点（可以自由选择）
- 再放置约束节点（必须映射到指定 SN 节点）

**优势**：
- 灵活性：非约束节点可以选择最优位置
- 约束满足：约束节点必须满足约束条件
- 资源优化：先优化非约束节点，再处理约束节点

---

## 六、完整算法伪代码

```
算法：GAL (Greedy Allocation Algorithm)

输入：
  - vn: 虚拟网络图（VN）
  - env: 环境对象（包含 SN 图和资源状态）

输出：
  - (success, mapping, r_t): 成功标志、节点映射、奖励

步骤：
  1. 计算 SN NodeRank（如果未计算）
     - H(u) = cpu * comm_bandwidth
     - 构建邻接矩阵
     - 计算前向概率 pF
     - 传播两轮并归一化
     - 做三次幂并归一化
  
  2. 分离约束节点和非约束节点
     - non_constraint_indices, constraint_indices, constraint_mapping = separate_constraint_nodes(vn)
  
  3. 贪心放置非约束节点
     - 对于每个非约束节点 vn_node：
       a. 计算资源需求（cpu, mem, disk）
       b. 找到所有满足资源需求的 SN 节点
       c. 按 NodeRank 降序排序
       d. 选择 NodeRank 最高的 SN 节点
       e. 临时扣减资源
     - 如果任何节点无法放置，返回失败
  
  4. 恢复临时扣减
  
  5. 放置约束节点
     - 对于每个约束节点：
       a. 检查指定的 SN 节点是否存在
       b. 检查资源是否足够
       c. 添加到映射
     - 如果任何约束节点无法放置，返回失败
  
  6. 验证映射并计算路径
     - vn_paths = env._compute_paths_and_bw_demand(vn, full_mapping)
     - 如果路径不存在，返回失败
  
  7. 应用映射（扣减资源）
     - env._apply_mapping(vn, full_mapping, vn_paths)
  
  8. 返回成功
```

---

## 七、代码结构

### 7.1 核心类和方法

**GreedyAllocator** (`baselines/GAL.py`)
- `__init__(env)`: 初始化，计算 SN NodeRank
- `greedy_place(vn)`: 执行贪心放置

**NodeRank 计算** (`baselines/noderank_utils.py`)
- `compute_sn_noderank_from_graph(G_sn)`: 从图计算 NodeRank
- `compute_sn_noderank_from_file(path)`: 从文件计算 NodeRank

**约束节点处理** (`tests/constraint_handler.py`)
- `separate_constraint_nodes(vn)`: 分离约束节点和非约束节点
- `place_constraint_nodes(env, vn, non_constraint_mapping, constraint_mapping)`: 放置约束节点

**策略包装** (`tests/tester_gal.py`)
- `GALPlacementStrategy`: 策略类，实现 `PlacementStrategy` 接口
- `place(vn, sn_state, env, context)`: 执行放置，返回 `StrategyResult`

### 7.2 调用流程

```
测试框架
  ↓
GALPlacementStrategy.place()
  ↓
GreedyAllocator.greedy_place()
  ↓
separate_constraint_nodes() → 分离节点
  ↓
贪心放置非约束节点（基于 NodeRank）
  ↓
place_constraint_nodes() → 放置约束节点
  ↓
env._compute_paths_and_bw_demand() → 计算路径
  ↓
env._apply_mapping() → 应用映射
  ↓
返回 StrategyResult
```

---

## 八、算法复杂度

### 8.1 时间复杂度

- **NodeRank 计算**：O(SN_nodes²)（一次计算，可缓存）
- **非约束节点放置**：O(VN_nodes × SN_nodes)
- **约束节点放置**：O(constraint_nodes)
- **路径计算**：O(VN_edges × log(SN_nodes))（最短路径）

**总体复杂度**：O(VN_nodes × SN_nodes + VN_edges × log(SN_nodes))

### 8.2 空间复杂度

- **NodeRank 数组**：O(SN_nodes)
- **临时映射**：O(VN_nodes)
- **临时资源扣减记录**：O(VN_nodes)

**总体空间复杂度**：O(SN_nodes + VN_nodes)

---

## 九、算法优化点

### 9.1 NodeRank 缓存

**优化**：
- NodeRank 在 `__init__` 时计算一次，后续复用
- 避免重复计算

**代码**：
```python
def __init__(self, env: SimuVNEEnv):
    self.env = env
    self.sn_noderank = compute_sn_noderank_from_graph(env.G_sn)  # 只计算一次
```

### 9.2 资源检查优化

**优化**：
- 只检查满足资源需求的 SN 节点
- 避免不必要的 NodeRank 查询

**代码**：
```python
if (res_cpu >= demand_cpu - 1e-9 and 
    res_mem >= demand_mem - 1e-9 and 
    res_disk >= demand_disk - 1e-9):
    # 只有满足资源需求的节点才加入候选列表
```

### 9.3 临时资源扣减

**优化**：
- 使用临时扣减记录，失败时可以回滚
- 避免部分成功导致的状态不一致

**代码**：
```python
temporary_deductions.append((best_sn, demand_cpu, demand_mem, demand_disk))
# 如果后续失败，可以回滚
_restore_temporary_deductions()
```

---

## 十、与其他算法的对比

### 10.1 GAL vs GA (Genetic Algorithm)

| 特性 | GAL | GA |
|------|-----|-----|
| 策略 | 贪心 | 进化算法 |
| 时间复杂度 | O(VN × SN) | O(population × generations × VN × SN) |
| 全局优化 | 否（局部最优） | 是（可能找到全局最优） |
| 计算速度 | 快 | 慢 |
| 适用场景 | 实时决策 | 离线优化 |

### 10.2 GAL vs GAL-SN (SN-Sorted)

| 特性 | GAL | GAL-SN |
|------|-----|--------|
| 排序依据 | NodeRank | SN 节点资源（降序） |
| 拓扑感知 | 是（NodeRank 考虑邻居） | 否（只考虑节点资源） |
| 计算复杂度 | 需要计算 NodeRank | 直接排序 |

### 10.3 GAL vs 神经网络策略

| 特性 | GAL | 神经网络策略 |
|------|-----|------------|
| 决策方式 | 规则（贪心） | 学习（神经网络） |
| 训练需求 | 无需训练 | 需要训练 |
| 适应性 | 固定策略 | 可适应不同场景 |
| 计算速度 | 快 | 中等（需要推理） |

---

## 十一、使用示例

### 11.1 基本使用

```python
from baselines.GAL import GreedyAllocator
from env import SimuVNEEnv

# 创建环境
env = SimuVNEEnv(sn_topology_path="topo/SN_topology_2.json")
env.reset()

# 创建 GAL 分配器
allocator = GreedyAllocator(env)

# 执行放置
success, mapping, r_t = allocator.greedy_place(vn)

if success:
    print(f"放置成功，映射: {mapping}")
else:
    print("放置失败")
```

### 11.2 在测试框架中使用

```python
from tests.tester_gal import GALPlacementStrategy

strategy = GALPlacementStrategy()
result = strategy.place(vn, sn_state, env, context=context)

if result.success:
    print(f"映射: {result.mapping}")
```

---

## 十二、总结

### 12.1 算法优势

1. **计算速度快**：时间复杂度低，适合实时决策
2. **实现简单**：代码清晰，易于理解和维护
3. **资源导向**：使用 NodeRank 优先选择资源丰富的节点
4. **拓扑感知**：NodeRank 考虑节点的邻居关系
5. **约束支持**：支持约束节点的两阶段放置

### 12.2 算法局限

1. **局部最优**：贪心策略可能陷入局部最优
2. **不考虑后续**：不考虑后续任务的影响
3. **资源竞争**：多个任务可能竞争相同的资源丰富节点

### 12.3 适用场景

- **实时决策**：需要快速响应的场景
- **资源丰富**：SN 资源相对充足的场景
- **简单场景**：不需要复杂优化的场景
- **基准对比**：作为其他算法的对比基准

---

**文档版本**：v1.0  
**创建时间**：2025年1月

