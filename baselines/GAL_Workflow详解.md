# GAL策略完整工作流程详解

## 一、概述

**GAL (Greedy Allocation Algorithm)** 是一个基于 NodeRank 的贪心放置算法，采用两阶段策略处理约束节点和非约束节点。

---

## 二、完整工作流程

### 阶段0：初始化（GreedyAllocator.__init__）

**代码位置**：`baselines/GAL.py` 第32-45行

**流程**：
```
1. 保存环境引用
   self.env = env

2. 创建SN节点索引映射
   - 获取所有SN节点并排序：sn_node_list = sorted(env.G_sn.nodes())
   - 创建映射：sn_node_to_idx = {node_id: idx for idx, node_id in enumerate(sn_node_list)}
   - 用途：将SN节点ID映射到noderank数组的索引

3. 计算SN NodeRank
   - 调用：compute_sn_noderank_from_graph(env.G_sn)
   - 结果：self.sn_noderank（numpy数组）
   - 特点：只计算一次，后续复用

4. 验证NodeRank长度
   - 检查：len(self.sn_noderank) == len(sn_node_list)
   - 如果不匹配，抛出异常
```

**关键点**：
- NodeRank在初始化时计算一次，后续不再更新
- 索引映射确保SN节点ID与noderank数组索引对应

---

### 阶段1：任务到达（测试框架调用）

**代码位置**：`tests/tester_gal.py` 第50-70行

**流程**：
```
1. 测试框架调用 GALPlacementStrategy.place()
   - 输入：vn（虚拟网络图）、sn_state、env、context

2. 创建环境副本
   temp_env = copy.deepcopy(env)
   - 原因：不直接修改主环境，避免影响其他策略

3. 创建GAL分配器
   allocator = GALAllocator(temp_env)
   - 在环境副本上创建分配器
   - 重新计算NodeRank（基于环境副本的SN图）

4. 调用贪心放置
   success, mapping, _ = allocator.greedy_place(vn)
```

---

### 阶段2：分离约束节点和非约束节点

**代码位置**：`baselines/GAL.py` 第60-61行

**流程**：
```
调用：separate_constraint_nodes(vn)

输入：
  - vn: 虚拟网络图（可能包含 constraint_nodes 属性）

处理：
  - 检查 vn.constraint_nodes 属性
  - 如果为 None 或不存在：所有节点都是非约束节点
  - 否则：
    * constraint_nodes[i] = None → 非约束节点
    * constraint_nodes[i] = sn_id → 约束节点，必须映射到 sn_id

输出：
  - non_constraint_indices: 非约束节点的VN索引列表
  - constraint_indices: 约束节点的VN索引列表
  - constraint_mapping: {vn_idx: sn_node_id} 约束节点映射字典
```

**示例**：
```python
# 假设VN有3个节点
vn.constraint_nodes = [None, 5, None]
# 结果：
# non_constraint_indices = [0, 2]
# constraint_indices = [1]
# constraint_mapping = {1: 5}
```

---

### 阶段3：计算非约束节点需求并排序

**代码位置**：`baselines/GAL.py` 第77-101行

**流程**：
```
1. 遍历所有非约束节点
   for vn_idx in non_constraint_indices:
       feats = vn.x[vn_idx]
       norm_cpu = feats[0]  # 归一化CPU需求
       norm_mem = feats[1]  # 归一化内存需求
       norm_disk = feats[2] # 归一化磁盘需求

2. 计算归一化需求总和
   norm_demand = norm_cpu + norm_mem + norm_disk

3. 转换为绝对需求
   abs_cpu = norm_cpu * (sn_max_capacity['cpu_max'] + 1e-8)
   abs_mem = norm_mem * (sn_max_capacity['mem_max'] + 1e-8)
   abs_disk = norm_disk * (sn_max_capacity['disk_max'] + 1e-8)

4. 构建需求列表
   vn_demands.append({
       'vn_node': vn_idx,
       'abs_cpu': abs_cpu,
       'abs_mem': abs_mem,
       'abs_disk': abs_disk,
       'norm_demand': norm_demand,
   })

5. 按归一化需求降序排序
   vn_demands.sort(key=lambda x: x['norm_demand'], reverse=True)
   # 需求大的节点优先放置
```

**关键点**：
- **需求大的节点优先放置**：减少资源冲突，提高接受率
- 使用归一化需求排序，确保不同资源类型的公平性

---

### 阶段4：贪心放置非约束节点

**代码位置**：`baselines/GAL.py` 第103-147行

**流程**：
```
对每个非约束节点（按排序后的顺序）：

1. 提取节点需求
   vn_node = vn_info['vn_node']
   demand_cpu = vn_info['abs_cpu']
   demand_mem = vn_info['abs_mem']
   demand_disk = vn_info['abs_disk']

2. 查找所有满足资源需求的SN节点
   sn_nodes_with_rank = []
   for sn_node in self.env.G_sn.nodes:
       # 获取noderank索引
       sn_idx = self.sn_node_to_idx.get(sn_node)
       
       # 检查资源是否满足
       if (res_cpu >= demand_cpu - 1e-9 and 
           res_mem >= demand_mem - 1e-9 and 
           res_disk >= demand_disk - 1e-9):
           # 获取noderank值
           noderank = self.sn_noderank[sn_idx]
           sn_nodes_with_rank.append({
               'sn_node': sn_node,
               'noderank': noderank,
           })

3. 检查是否有可用节点
   if not sn_nodes_with_rank:
       # 回滚所有临时扣减
       _restore_temporary_deductions()
       return False, {}, penalty

4. 按noderank降序排序，选择最高的
   sn_nodes_with_rank.sort(key=lambda x: x['noderank'], reverse=True)
   best_sn = sn_nodes_with_rank[0]['sn_node']
   non_constraint_mapping[vn_node] = best_sn

5. 临时扣减资源
   nd = self.env.G_sn.nodes[best_sn]
   nd['cpu_res'] -= demand_cpu
   nd['mem_res'] -= demand_mem
   nd['disk_res'] -= demand_disk
   temporary_deductions.append((best_sn, demand_cpu, demand_mem, demand_disk))
```

**关键点**：
- **选择noderank最高的可用SN节点**：优先使用资源丰富、拓扑位置好的节点
- **临时扣减资源**：确保后续节点选择时考虑已占用的资源
- **失败回滚**：如果任何节点无法放置，回滚所有临时扣减

---

### 阶段5：恢复临时扣减

**代码位置**：`baselines/GAL.py` 第151行

**流程**：
```
调用：_restore_temporary_deductions()

作用：
  - 恢复所有临时扣减的资源
  - 原因：后续会统一通过 env._apply_mapping 应用映射
  - 确保资源状态一致
```

**为什么需要恢复**：
- 临时扣减只是为了检查资源是否足够
- 恢复后，约束节点放置时看到的是原始资源状态
- 最后统一应用映射，避免重复扣减

---

### 阶段6：放置约束节点

**代码位置**：`baselines/GAL.py` 第153-159行

**流程**：
```
调用：place_constraint_nodes(env, vn, non_constraint_mapping, constraint_mapping)

处理：
  1. 构建完整映射（包含非约束节点映射）
     full_mapping = dict(non_constraint_mapping)

  2. 对每个约束节点：
     a. 检查指定的SN节点是否存在
     b. 检查资源是否足够（考虑非约束节点已占用的资源）
     c. 如果满足，添加到映射
     d. 如果不满足，返回失败

  3. 返回结果
     - success: 是否成功
     - full_mapping: 完整映射（包含所有节点）
     - failure_reason: 失败原因（如果失败）

检查失败：
  if not success:
      return False, {}, penalty
```

**关键点**：
- 约束节点必须映射到指定的SN节点
- 资源检查考虑非约束节点已占用的资源（通过环境副本的状态）

---

### 阶段7：验证映射并计算路径

**代码位置**：`baselines/GAL.py` 第161-164行

**流程**：
```
调用：env._compute_paths_and_bw_demand(vn, full_mapping)

处理：
  1. 对每条VN链路：
     - 获取源节点和目标节点映射的SN节点
     - 如果映射到同一SN节点：路径为 [sn_node]
     - 否则：计算最短路径（Dijkstra算法）

  2. 检查路径是否存在
     - 如果任何路径不存在，返回 None

  3. 返回路径列表
     vn_paths = [(sn_u, sn_v, path), ...]

检查失败：
  if vn_paths is None:
      return False, {}, penalty
```

**关键点**：
- 使用最短路径算法计算VN链路的物理路径
- 如果路径不存在，放置失败

---

### 阶段8：应用映射（统一扣减资源）

**代码位置**：`baselines/GAL.py` 第166-167行

**流程**：
```
调用：env._apply_mapping(vn, full_mapping, vn_paths)

处理：
  1. 按VN节点需求从高到低排序
  2. 依次扣减节点资源（CPU、内存、磁盘）
  3. 扣减链路带宽资源（基于路径）

结果：
  - SN节点的剩余资源被更新
  - SN链路的剩余带宽被更新
```

**关键点**：
- 统一应用映射，确保资源状态一致
- 按需求排序扣减，避免资源冲突

---

### 阶段9：返回结果

**代码位置**：`baselines/GAL.py` 第169-170行

**流程**：
```
返回：True, full_mapping, 0.0

说明：
  - success = True：放置成功
  - mapping = full_mapping：完整映射（包含所有节点）
  - r_t = 0.0：r_t在外部计算（基于当前所有存活的workflow）
```

---

## 三、在测试框架中的集成

### 测试框架调用流程

**代码位置**：`tests/tester_gal.py` 和 `tests/test_strategy.py`

**流程**：
```
1. 测试框架调用 GALPlacementStrategy.place()
   ↓
2. 创建环境副本
   temp_env = copy.deepcopy(env)
   ↓
3. 创建GAL分配器
   allocator = GALAllocator(temp_env)
   ↓
4. 调用贪心放置
   success, mapping, _ = allocator.greedy_place(vn)
   ↓
5. 返回StrategyResult
   StrategyResult(success=success, mapping=mapping, metadata={...})
   ↓
6. 测试框架处理结果
   - 如果成功：添加到active_workflows
   - 如果失败：记录失败原因
   - 计算r_t（基于所有存活的workflow）
```

---

## 四、关键决策点

### 决策点1：VN节点排序

**决策**：按归一化需求从大到小排序

**原因**：
- 需求大的节点优先放置，减少资源冲突
- 提高接受率

**实现**：
```python
vn_demands.sort(key=lambda x: x['norm_demand'], reverse=True)
```

### 决策点2：SN节点选择

**决策**：选择noderank最高的可用SN节点

**原因**：
- noderank反映节点的资源丰富度和拓扑重要性
- 优先使用资源丰富、位置好的节点

**实现**：
```python
sn_nodes_with_rank.sort(key=lambda x: x['noderank'], reverse=True)
best_sn = sn_nodes_with_rank[0]['sn_node']
```

### 决策点3：资源检查

**决策**：检查CPU、内存、磁盘是否都满足需求

**原因**：
- 确保所有资源类型都足够
- 避免部分资源不足导致的失败

**实现**：
```python
if (res_cpu >= demand_cpu - 1e-9 and 
    res_mem >= demand_mem - 1e-9 and 
    res_disk >= demand_disk - 1e-9):
    # 满足需求
```

---

## 五、资源管理流程

### 临时资源扣减

**目的**：在放置过程中跟踪资源使用

**流程**：
```
1. 放置非约束节点时临时扣减资源
   nd['cpu_res'] -= demand_cpu
   temporary_deductions.append((sn_node, cpu, mem, disk))

2. 如果后续节点无法放置，回滚所有扣减
   _restore_temporary_deductions()

3. 如果所有节点都成功放置，恢复临时扣减
   _restore_temporary_deductions()
   （因为后续会统一应用映射）
```

### 最终资源应用

**目的**：统一应用映射，更新资源状态

**流程**：
```
1. 放置约束节点后，验证映射
2. 计算路径
3. 统一应用映射（扣减资源）
   env._apply_mapping(vn, full_mapping, vn_paths)
```

---

## 六、失败处理

### 失败场景1：非约束节点无法放置

**处理**：
```
1. 回滚所有临时扣减
   _restore_temporary_deductions()

2. 返回失败
   return False, {}, penalty
```

### 失败场景2：约束节点无法放置

**处理**：
```
1. place_constraint_nodes 返回失败
2. 返回失败
   return False, {}, penalty
```

### 失败场景3：路径不存在

**处理**：
```
1. _compute_paths_and_bw_demand 返回 None
2. 返回失败
   return False, {}, penalty
```

---

## 七、完整流程图

```
初始化阶段：
  GreedyAllocator.__init__(env)
    ↓
  创建SN节点索引映射
    ↓
  计算SN NodeRank（一次）
    ↓
  验证NodeRank长度

任务到达：
  测试框架调用 GALPlacementStrategy.place()
    ↓
  创建环境副本
    ↓
  创建GAL分配器（重新计算NodeRank）
    ↓
  调用 greedy_place(vn)

放置流程：
  1. 分离约束节点和非约束节点
    ↓
  2. 计算非约束节点需求并排序（需求大的优先）
    ↓
  3. 对每个非约束节点（按排序顺序）：
     a. 查找所有满足资源需求的SN节点
     b. 按noderank降序排序
     c. 选择noderank最高的SN节点
     d. 临时扣减资源
    ↓
  4. 恢复临时扣减
    ↓
  5. 放置约束节点（检查资源并添加到映射）
    ↓
  6. 验证映射并计算路径
    ↓
  7. 应用映射（统一扣减资源）
    ↓
  8. 返回成功

测试框架处理：
  如果成功：
    - 添加到active_workflows
    - 计算r_t
    - 记录结果
  如果失败：
    - 记录失败原因
    - 不添加到active_workflows
```

---

## 八、关键特性总结

### 1. 两阶段放置策略
- **阶段1**：非约束节点（按需求排序，选择noderank最高的SN节点）
- **阶段2**：约束节点（映射到指定的SN节点）

### 2. NodeRank指导
- 使用SN NodeRank评估节点重要性
- 优先选择noderank高的SN节点

### 3. 需求优先
- VN节点按需求从大到小排序
- 需求大的节点优先放置

### 4. 资源管理
- 临时扣减用于检查资源是否足够
- 统一应用映射，确保资源状态一致

### 5. 失败回滚
- 如果任何节点无法放置，回滚所有临时扣减
- 确保资源状态一致性

---

**文档版本**：v1.0  
**创建时间**：2025年1月

