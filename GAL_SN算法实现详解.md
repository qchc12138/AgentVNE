# GAL-SN (Greedy Allocation Algorithm - SN-Sorted) 算法实现详解

## 一、算法概述

**GAL-SN**（现在改名为 **greedy**）是一种基于 SN 节点预排序的贪心放置算法。

**核心思想**：
1. **SN 节点预排序**：对 SN 节点按资源总量（归一化后的 CPU + 内存 + 磁盘）从高到低排序，**只排序一次**
2. **VN 节点依次选择**：VN 节点按顺序从排序后的 SN 节点列表中选择，从资源总量高的开始
3. **立即扣减资源**：每次选择后立即扣减资源，影响后续 VN 节点的选择

**与其他 GAL 变体的区别**：
- **GAL**：对 VN 节点排序，为每个 VN 节点选择 NodeRank 最高的 SN 节点
- **GAL-PN**：对 VN 节点排序，逐节点贪心选择并立即扣减资源
- **GAL-SN**：**仅对 SN 节点排序一次**，VN 节点依次从排序后的 SN 节点中选择

---

## 二、算法架构

### 2.1 主要组件

1. **GreedyAllocator** (`baselines/GAL_3.py`)
   - SN 节点预排序
   - VN 节点依次选择

2. **约束节点处理** (`tests/constraint_handler.py`)
   - 两阶段放置策略

3. **策略包装** (`tests/tester_gal_3.py`)
   - 将 GAL-SN 算法包装为统一的策略接口

---

## 三、SN 节点排序

### 3.1 排序方法

**代码位置**：`baselines/GAL_3.py` 第30-62行

```python
def _sort_sn_nodes(self) -> list:
    """
    对SN节点按资源总量从高到低排序（仅排序一次）。
    
    Returns:
        排序后的SN节点ID列表
    """
    if self._sn_sorted_list is not None:
        return self._sn_sorted_list  # 使用缓存
    
    sn_nodes = []
    for sn_node in self.env.G_sn.nodes:
        nd = self.env.G_sn.nodes[sn_node]
        res_cpu = nd['cpu_res']
        res_mem = nd['mem_res']
        res_disk = nd['disk_res']
        
        # 计算资源总量（归一化）
        norm_res_cpu = res_cpu / (self.env._sn_max_capacity['cpu_max'] + 1e-8)
        norm_res_mem = res_mem / (self.env._sn_max_capacity['mem_max'] + 1e-8)
        norm_res_disk = res_disk / (self.env._sn_max_capacity['disk_max'] + 1e-8)
        total_res = norm_res_cpu + norm_res_mem + norm_res_disk
        
        sn_nodes.append({
            'sn_node': sn_node,
            'total_res': total_res,
        })
    
    # 按资源总量从高到低排序
    sn_nodes.sort(key=lambda x: x['total_res'], reverse=True)
    self._sn_sorted_list = [item['sn_node'] for item in sn_nodes]
    
    return self._sn_sorted_list
```

### 3.2 排序特点

1. **归一化资源**：
   - 将 CPU、内存、磁盘资源归一化到 [0, 1] 范围
   - 使用 SN 最大容量进行归一化

2. **资源总量**：
   - `total_res = norm_res_cpu + norm_res_mem + norm_res_disk`
   - 简单相加，不考虑权重

3. **排序缓存**：
   - 排序结果缓存在 `self._sn_sorted_list`
   - 如果资源状态变化，需要重置缓存

### 3.3 排序重置

**代码位置**：`baselines/GAL_3.py` 第64-66行

```python
def _reset_sn_sort(self) -> None:
    """重置SN节点排序缓存（当资源发生变化时调用）。"""
    self._sn_sorted_list = None
```

**调用时机**：
- 每次 `greedy_place()` 开始时重置（因为资源状态可能已变化）
- 放置成功后重置（因为资源已发生变化）

---

## 四、GAL-SN 贪心放置算法

### 4.1 算法流程

**代码位置**：`baselines/GAL_3.py` 第68-167行

#### 阶段1：分离约束节点和非约束节点

```python
from tests.constraint_handler import separate_constraint_nodes, place_constraint_nodes

non_constraint_indices, constraint_indices, constraint_mapping = separate_constraint_nodes(vn)
```

**说明**：
- 分离约束节点（必须映射到指定 SN 节点）和非约束节点（可以自由选择）
- `constraint_mapping`：约束节点的映射关系 `{vn_idx: sn_node_id}`

#### 阶段2：重置排序缓存并获取排序后的 SN 节点列表

```python
self._reset_sn_sort()  # 重置缓存（因为资源状态可能已变化）
sn_sorted_list = self._sort_sn_nodes()  # 获取排序后的SN节点列表（按资源总量从高到低）
```

**说明**：
- 每次放置前重置排序缓存，确保使用最新的资源状态
- 获取按资源总量从高到低排序的 SN 节点列表

#### 阶段3：VN 节点依次选择 SN 节点

```python
non_constraint_mapping = {}
temporary_deductions = []  # 记录临时扣减的资源，用于失败时回滚

for vn_idx in non_constraint_indices:
    # 计算VN节点的资源需求
    feats = vn.x[vn_idx]
    norm_cpu = float(feats[0].item())
    norm_mem = float(feats[1].item())
    norm_disk = float(feats[2].item())
    
    abs_cpu = norm_cpu * (self.env._sn_max_capacity['cpu_max'] + 1e-8)
    abs_mem = norm_mem * (self.env._sn_max_capacity['mem_max'] + 1e-8)
    abs_disk = norm_disk * (self.env._sn_max_capacity['disk_max'] + 1e-8)
    
    # 从排序后的SN节点列表中选择第一个满足资源需求的节点
    placed = False
    for sn_node in sn_sorted_list:
        nd = self.env.G_sn.nodes[sn_node]
        res_cpu = nd['cpu_res']  # 当前剩余资源（已反映之前的扣减）
        res_mem = nd['mem_res']
        res_disk = nd['disk_res']
        
        # 检查是否满足需求
        if (res_cpu >= abs_cpu - 1e-9 and 
            res_mem >= abs_mem - 1e-9 and 
            res_disk >= abs_disk - 1e-9):
            # 找到合适的SN节点，进行映射
            non_constraint_mapping[vn_idx] = sn_node
            
            # 立即扣减该SN节点的资源（影响后续VN节点的选择）
            nd['cpu_res'] -= abs_cpu
            nd['mem_res'] -= abs_mem
            nd['disk_res'] -= abs_disk
            
            # 记录扣减信息（用于失败时回滚）
            temporary_deductions.append((sn_node, abs_cpu, abs_mem, abs_disk))
            
            placed = True
            break
    
    # 如果没有找到合适的SN节点，放置失败，回滚所有资源
    if not placed:
        # 回滚之前的资源扣减
        for sn_node, cpu, mem, disk in temporary_deductions:
            nd = self.env.G_sn.nodes[sn_node]
            nd['cpu_res'] += cpu
            nd['mem_res'] += mem
            nd['disk_res'] += disk
        return False, {}, self.env.penalty
```

**关键点**：
1. **依次选择**：VN 节点按顺序（`non_constraint_indices` 的顺序）选择 SN 节点
2. **从高到低**：从排序后的 SN 节点列表的第一个开始（资源总量最高的）
3. **第一个满足需求**：选择第一个满足资源需求的 SN 节点
4. **立即扣减**：选择后立即扣减资源，影响后续 VN 节点的选择
5. **失败回滚**：如果任何 VN 节点无法放置，回滚所有资源扣减

#### 阶段4：恢复临时扣减

```python
# 恢复临时扣减的资源（因为后续会统一应用映射）
for sn_node, cpu, mem, disk in temporary_deductions:
    nd = self.env.G_sn.nodes[sn_node]
    nd['cpu_res'] += cpu
    nd['mem_res'] += mem
    nd['disk_res'] += disk
```

**说明**：
- 恢复临时扣减的资源
- 因为后续会统一通过 `env._apply_mapping` 应用映射

#### 阶段5：放置约束节点

```python
success, full_mapping, failure_reason = place_constraint_nodes(
    self.env, vn, non_constraint_mapping, constraint_mapping
)

if not success:
    return False, {}, self.env.penalty
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

# 重置SN节点排序缓存（因为资源已发生变化）
self._reset_sn_sort()

# 返回成功
return True, full_mapping, 0.0  # r_t在外部计算
```

---

## 五、算法特点

### 5.1 SN 节点预排序

**特点**：
- **只排序一次**：在每次放置开始时排序一次
- **资源导向**：按资源总量从高到低排序
- **简单高效**：排序复杂度 O(SN_nodes × log(SN_nodes))

**优势**：
- 计算速度快：排序一次，后续直接使用
- 资源优先：优先使用资源丰富的节点
- 实现简单：不需要复杂的 NodeRank 计算

**局限**：
- 不考虑拓扑结构：只考虑节点资源，不考虑邻居关系
- 静态排序：排序基于当前资源状态，不考虑动态变化

### 5.2 VN 节点依次选择

**特点**：
- VN 节点按顺序（`non_constraint_indices` 的顺序）选择 SN 节点
- 从排序后的 SN 节点列表的第一个开始（资源总量最高的）
- 选择第一个满足资源需求的 SN 节点

**优势**：
- 简单直接：不需要为每个 VN 节点重新排序 SN 节点
- 资源优先：优先使用资源丰富的节点
- 计算效率高：时间复杂度 O(VN_nodes × SN_nodes)

**局限**：
- 顺序依赖：VN 节点的顺序会影响放置结果
- 局部最优：每次选择局部最优，不考虑全局最优

### 5.3 立即资源扣减

**特点**：
- 每次选择后立即扣减资源
- 影响后续 VN 节点的选择
- 如果失败，回滚所有资源扣减

**优势**：
- 实时反映资源状态：后续选择基于最新的资源状态
- 避免资源冲突：确保资源不被重复分配

**局限**：
- 顺序敏感：VN 节点的顺序会影响资源分配
- 可能陷入局部最优：早期节点可能占用过多资源

---

## 六、完整算法伪代码

```
算法：GAL-SN (Greedy Allocation Algorithm - SN-Sorted)

输入：
  - vn: 虚拟网络图（VN）
  - env: 环境对象（包含 SN 图和资源状态）

输出：
  - (success, mapping, r_t): 成功标志、节点映射、奖励

步骤：
  1. 分离约束节点和非约束节点
     - non_constraint_indices, constraint_indices, constraint_mapping = separate_constraint_nodes(vn)
  
  2. 重置排序缓存并获取排序后的 SN 节点列表
     - self._reset_sn_sort()
     - sn_sorted_list = self._sort_sn_nodes()  # 按资源总量从高到低排序
  
  3. VN 节点依次选择 SN 节点
     - 对于每个非约束节点 vn_idx（按顺序）：
       a. 计算资源需求（cpu, mem, disk）
       b. 从排序后的 SN 节点列表的第一个开始遍历
       c. 选择第一个满足资源需求的 SN 节点
       d. 立即扣减该 SN 节点的资源
       e. 记录临时扣减信息
     - 如果任何节点无法放置，回滚所有资源扣减，返回失败
  
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
  
  8. 重置 SN 节点排序缓存
  
  9. 返回成功
```

---

## 七、代码结构

### 7.1 核心类和方法

**GreedyAllocator** (`baselines/GAL_3.py`)
- `__init__(env)`: 初始化
- `_sort_sn_nodes()`: 对 SN 节点按资源总量排序（缓存）
- `_reset_sn_sort()`: 重置排序缓存
- `greedy_place(vn)`: 执行贪心放置

**约束节点处理** (`tests/constraint_handler.py`)
- `separate_constraint_nodes(vn)`: 分离约束节点和非约束节点
- `place_constraint_nodes(env, vn, non_constraint_mapping, constraint_mapping)`: 放置约束节点

**策略包装** (`tests/tester_gal_3.py`)
- `GAL3PlacementStrategy`: 策略类，实现 `PlacementStrategy` 接口
- `place(vn, sn_state, env, context)`: 执行放置，返回 `StrategyResult`

### 7.2 调用流程

```
测试框架
  ↓
GAL3PlacementStrategy.place()
  ↓
GreedyAllocator.greedy_place()
  ↓
separate_constraint_nodes() → 分离节点
  ↓
_sort_sn_nodes() → SN节点排序（按资源总量）
  ↓
VN节点依次选择SN节点（从排序列表的第一个开始）
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

- **SN 节点排序**：O(SN_nodes × log(SN_nodes))（每次放置时排序一次）
- **VN 节点选择**：O(VN_nodes × SN_nodes)（最坏情况，每个 VN 节点遍历所有 SN 节点）
- **约束节点放置**：O(constraint_nodes)
- **路径计算**：O(VN_edges × log(SN_nodes))（最短路径）

**总体复杂度**：O(SN_nodes × log(SN_nodes) + VN_nodes × SN_nodes + VN_edges × log(SN_nodes))

**平均情况**：如果 SN 节点资源充足，每个 VN 节点通常只需要检查前几个 SN 节点，复杂度接近 O(VN_nodes × log(SN_nodes))

### 8.2 空间复杂度

- **排序列表**：O(SN_nodes)
- **临时映射**：O(VN_nodes)
- **临时资源扣减记录**：O(VN_nodes)

**总体空间复杂度**：O(SN_nodes + VN_nodes)

---

## 九、算法优化点

### 9.1 排序缓存

**优化**：
- 排序结果缓存在 `self._sn_sorted_list`
- 避免重复排序

**代码**：
```python
if self._sn_sorted_list is not None:
    return self._sn_sorted_list  # 使用缓存
```

**注意**：
- 资源状态变化时需要重置缓存
- 每次 `greedy_place()` 开始时重置缓存

### 9.2 资源检查优化

**优化**：
- 只检查满足资源需求的 SN 节点
- 从资源总量高的开始检查，提高命中率

**代码**：
```python
for sn_node in sn_sorted_list:  # 从资源总量高的开始
    if (res_cpu >= abs_cpu - 1e-9 and 
        res_mem >= abs_mem - 1e-9 and 
        res_disk >= abs_disk - 1e-9):
        # 找到合适的节点，立即返回
        break
```

### 9.3 临时资源扣减

**优化**：
- 使用临时扣减记录，失败时可以回滚
- 避免部分成功导致的状态不一致

**代码**：
```python
temporary_deductions.append((sn_node, abs_cpu, abs_mem, abs_disk))
# 如果后续失败，可以回滚
for sn_node, cpu, mem, disk in temporary_deductions:
    nd['cpu_res'] += cpu
    nd['mem_res'] += mem
    nd['disk_res'] += disk
```

---

## 十、与其他算法的对比

### 10.1 GAL-SN vs GAL

| 特性 | GAL-SN | GAL |
|------|--------|-----|
| SN 节点排序 | 按资源总量排序 | 使用 NodeRank |
| 排序次数 | 一次（每次放置） | 一次（初始化时） |
| 拓扑感知 | 否 | 是（NodeRank 考虑邻居） |
| VN 节点选择 | 依次选择 | 按需求排序后选择 |
| 计算复杂度 | O(SN × log(SN) + VN × SN) | O(VN × SN) |

### 10.2 GAL-SN vs GAL-PN

| 特性 | GAL-SN | GAL-PN |
|------|--------|--------|
| SN 节点排序 | 按资源总量排序 | 不排序 |
| VN 节点排序 | 不排序 | 按需求排序 |
| 选择策略 | 依次从排序列表选择 | 为每个 VN 节点选择资源最多的 SN 节点 |
| 资源扣减 | 立即扣减 | 立即扣减 |

### 10.3 GAL-SN vs 神经网络策略

| 特性 | GAL-SN | 神经网络策略 |
|------|--------|------------|
| 决策方式 | 规则（贪心） | 学习（神经网络） |
| 训练需求 | 无需训练 | 需要训练 |
| 适应性 | 固定策略 | 可适应不同场景 |
| 计算速度 | 快 | 中等（需要推理） |
| 全局优化 | 否（局部最优） | 可能（通过训练） |

---

## 十一、使用示例

### 11.1 基本使用

```python
from baselines.GAL_3 import GreedyAllocator
from env import SimuVNEEnv

# 创建环境
env = SimuVNEEnv(sn_topology_path="topo/SN_topology_2.json")
env.reset()

# 创建 GAL-SN 分配器
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
from tests.tester_gal_3 import GAL3PlacementStrategy

strategy = GAL3PlacementStrategy()
result = strategy.place(vn, sn_state, env, context=context)

if result.success:
    print(f"映射: {result.mapping}")
```

---

## 十二、算法示例

### 12.1 简单示例

**场景**：
- SN 节点：SN1（资源总量=3.0），SN2（资源总量=2.0），SN3（资源总量=1.0）
- VN 节点：VN1（需求=1.0），VN2（需求=1.5），VN3（需求=0.5）

**步骤1：SN 节点排序**
```
sn_sorted_list = [SN1, SN2, SN3]  # 按资源总量从高到低
```

**步骤2：VN 节点依次选择**
```
VN1 → 选择 SN1（第一个满足需求）
  - SN1 资源：3.0 → 2.0

VN2 → 选择 SN1（第一个满足需求）
  - SN1 资源：2.0 → 0.5

VN3 → 选择 SN2（SN1 资源不足，选择 SN2）
  - SN2 资源：2.0 → 1.5

最终映射：{VN1: SN1, VN2: SN1, VN3: SN2}
```

### 12.2 复杂示例（考虑资源竞争）

**场景**：
- SN 节点：SN1（资源总量=2.0），SN2（资源总量=1.5），SN3（资源总量=1.0）
- VN 节点：VN1（需求=1.5），VN2（需求=1.0），VN3（需求=0.5）

**步骤1：SN 节点排序**
```
sn_sorted_list = [SN1, SN2, SN3]
```

**步骤2：VN 节点依次选择**
```
VN1 → 选择 SN1（第一个满足需求）
  - SN1 资源：2.0 → 0.5

VN2 → 选择 SN2（SN1 资源不足，选择 SN2）
  - SN2 资源：1.5 → 0.5

VN3 → 选择 SN1（SN1 资源足够）
  - SN1 资源：0.5 → 0.0

最终映射：{VN1: SN1, VN2: SN2, VN3: SN1}
```

**注意**：
- VN 节点的顺序会影响结果
- 如果 VN2 先选择，可能会选择 SN1，导致 VN1 无法放置

---

## 十三、总结

### 13.1 算法优势

1. **计算速度快**：排序一次，后续直接使用
2. **实现简单**：代码清晰，易于理解和维护
3. **资源导向**：优先使用资源丰富的节点
4. **实时反映资源状态**：立即扣减资源，影响后续选择

### 13.2 算法局限

1. **顺序依赖**：VN 节点的顺序会影响放置结果
2. **局部最优**：贪心策略可能陷入局部最优
3. **不考虑拓扑**：只考虑节点资源，不考虑邻居关系
4. **静态排序**：排序基于当前资源状态，不考虑动态变化

### 13.3 适用场景

- **资源丰富**：SN 资源相对充足的场景
- **简单场景**：不需要复杂优化的场景
- **实时决策**：需要快速响应的场景
- **基准对比**：作为其他算法的对比基准

---

**文档版本**：v1.0  
**创建时间**：2025年1月

