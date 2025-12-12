# GAL策略实现说明

## 一、算法概述

**GAL (Greedy Allocation Algorithm)** 是一个基于 NodeRank 的贪心放置算法，采用以下策略：

1. **动态NodeRank计算**：每次workflow到达时，基于SN节点的**剩余资源**重新计算NodeRank
2. **资源消耗优先**：从资源消耗最高的VN节点开始放置
3. **广度优先遍历**：按广度优先的方式选择下一个放置的VN节点
4. **NodeRank指导**：优先放在NodeRank最高的可用SN节点上

---

## 二、完整工作流程

### 阶段1：Workflow到达

**触发**：测试框架调用 `GALPlacementStrategy.place()`

**处理**：
```
1. 创建环境副本（不修改主环境）
   temp_env = copy.deepcopy(env)

2. 创建GAL分配器
   allocator = GALAllocator(temp_env)

3. 调用贪心放置
   success, mapping, _ = allocator.greedy_place(vn)
```

---

### 阶段2：分离约束节点和非约束节点

**代码位置**：`baselines/GAL.py` 第132-133行

**处理**：
```
调用：separate_constraint_nodes(vn)

结果：
  - non_constraint_indices: 非约束节点索引列表
  - constraint_indices: 约束节点索引列表
  - constraint_mapping: 约束节点映射字典
```

---

### 阶段3：基于剩余资源计算SN NodeRank

**代码位置**：`baselines/GAL.py` 第135-136行

**关键特性**：
- **每次workflow到达时重新计算**
- **基于剩余资源**（cpu_res, mem_res, disk_res），而不是初始资源

**处理**：
```
调用：sn_noderank = self._compute_noderank_from_residual_resources()

内部实现：
  compute_sn_noderank_from_graph(self.env.G_sn, use_residual_resources=True)
  
计算方式：
  H(u) = cpu_res * comm_bandwidth
  （使用剩余CPU资源，而不是初始CPU资源）
```

**优势**：
- NodeRank反映当前资源状态
- 资源被占用后，NodeRank会动态调整
- 优先使用资源丰富的节点

---

### 阶段4：找到资源消耗最高的VN节点

**代码位置**：`baselines/GAL.py` 第138-150行

**处理**：
```
遍历所有非约束节点：
  for vn_idx in non_constraint_indices:
      demand_info = self._get_vn_node_demand(vn, vn_idx)
      norm_demand = norm_cpu + norm_mem + norm_disk
      
      如果 norm_demand > max_demand:
          max_demand = norm_demand
          max_demand_node = vn_idx

结果：
  max_demand_node: 资源消耗最高的VN节点（作为BFS起始节点）
```

**关键点**：
- 使用归一化需求总和（norm_cpu + norm_mem + norm_disk）
- 需求大的节点优先放置，减少资源冲突

---

### 阶段5：广度优先遍历VN图

**代码位置**：`baselines/GAL.py` 第66-111行，第152-153行

**处理**：
```
调用：vn_placement_order = self._bfs_vn_nodes(vn, max_demand_node, non_constraint_indices)

BFS流程：
  1. 构建VN图的邻接列表
  2. 从max_demand_node开始BFS遍历
  3. 只收集非约束节点
  4. 如果还有未访问的非约束节点（孤立节点），添加到末尾

结果：
  vn_placement_order: 按BFS顺序排列的非约束节点索引列表
```

**示例**：
```
假设VN图：
  0 -- 1 -- 2
  |    |
  3    4

非约束节点：[0, 1, 2, 3, 4]
假设节点2资源消耗最高（max_demand_node = 2）

BFS顺序（从节点2开始）：
  2 → 1 → 0, 4 → 3

最终顺序：[2, 1, 0, 4, 3]
```

**优势**：
- 保持VN图的拓扑结构
- 相邻节点优先放置，可能减少路径长度
- 从资源消耗最高的节点开始，确保关键节点优先

---

### 阶段6：按BFS顺序贪心放置非约束节点

**代码位置**：`baselines/GAL.py` 第169-213行

**处理**：
```
对每个VN节点（按BFS顺序）：

1. 获取节点需求
   demand_info = self._get_vn_node_demand(vn, vn_node)
   demand_cpu, demand_mem, demand_disk

2. 查找所有满足资源需求的SN节点
   sn_nodes_with_rank = []
   for sn_node in self.env.G_sn.nodes:
       if 资源满足需求:
           noderank = sn_noderank[sn_idx]
           sn_nodes_with_rank.append({
               'sn_node': sn_node,
               'noderank': noderank,
           })

3. 检查是否有可用节点
   if not sn_nodes_with_rank:
       回滚所有临时扣减
       return False, {}, penalty

4. 按noderank降序排序，选择最高的
   sn_nodes_with_rank.sort(key=lambda x: x['noderank'], reverse=True)
   best_sn = sn_nodes_with_rank[0]['sn_node']
   non_constraint_mapping[vn_node] = best_sn

5. 临时扣减资源
   nd['cpu_res'] -= demand_cpu
   temporary_deductions.append((best_sn, demand_cpu, demand_mem, demand_disk))
```

**关键点**：
- **优先选择noderank最高的SN节点**：资源丰富、拓扑位置好
- **临时扣减资源**：确保后续节点选择时考虑已占用的资源
- **失败回滚**：如果任何节点无法放置，回滚所有临时扣减

---

### 阶段7：恢复临时扣减

**代码位置**：`baselines/GAL.py` 第215-216行

**处理**：
```
调用：_restore_temporary_deductions()

作用：
  - 恢复所有临时扣减的资源
  - 原因：后续会统一通过 env._apply_mapping 应用映射
```

---

### 阶段8：放置约束节点

**代码位置**：`baselines/GAL.py` 第218-224行

**处理**：
```
调用：place_constraint_nodes(env, vn, non_constraint_mapping, constraint_mapping)

处理：
  - 在非约束节点映射的基础上，放置约束节点
  - 约束节点必须映射到指定的SN节点
  - 检查资源是否足够（考虑非约束节点已占用的资源）

如果失败：
  return False, {}, penalty
```

---

### 阶段9：验证映射并计算路径

**代码位置**：`baselines/GAL.py` 第226-229行

**处理**：
```
调用：vn_paths = env._compute_paths_and_bw_demand(vn, full_mapping)

处理：
  - 对每条VN链路，计算最短路径
  - 如果路径不存在，返回 None

如果失败：
  return False, {}, penalty
```

---

### 阶段10：应用映射（统一扣减资源）

**代码位置**：`baselines/GAL.py` 第231-232行

**处理**：
```
调用：env._apply_mapping(vn, full_mapping, vn_paths)

处理：
  - 统一扣减节点资源（CPU、内存、磁盘）
  - 统一扣减链路带宽资源
  - 更新SN资源状态
```

---

### 阶段11：返回结果

**代码位置**：`baselines/GAL.py` 第234-235行

**处理**：
```
返回：True, full_mapping, 0.0

说明：
  - success = True：放置成功
  - mapping = full_mapping：完整映射
  - r_t = 0.0：r_t在外部计算
```

---

## 三、关键特性

### 1. 动态NodeRank计算

**特点**：
- 每次workflow到达时重新计算
- 基于剩余资源（cpu_res），而不是初始资源（cpu）
- NodeRank反映当前资源状态

**优势**：
- 资源被占用后，NodeRank会动态调整
- 优先使用资源丰富的节点
- 适应动态环境

### 2. 资源消耗优先

**特点**：
- 从资源消耗最高的VN节点开始放置
- 使用归一化需求总和（norm_cpu + norm_mem + norm_disk）

**优势**：
- 需求大的节点优先放置，减少资源冲突
- 提高接受率

### 3. 广度优先遍历

**特点**：
- 从资源消耗最高的节点开始BFS
- 保持VN图的拓扑结构
- 相邻节点优先放置

**优势**：
- 可能减少路径长度
- 保持VN图的连通性
- 更好的拓扑感知

### 4. NodeRank指导

**特点**：
- 优先选择noderank最高的可用SN节点
- noderank反映节点的资源丰富度和拓扑重要性

**优势**：
- 优先使用资源丰富、位置好的节点
- 提高资源利用率

---

## 四、完整流程图

```
Workflow到达
  ↓
创建环境副本和GAL分配器
  ↓
分离约束节点和非约束节点
  ↓
基于剩余资源重新计算SN NodeRank ← 关键：每次重新计算
  ↓
找到资源消耗最高的VN节点（作为BFS起始节点）
  ↓
按广度优先遍历VN图，获取非约束节点放置顺序
  ↓
对每个VN节点（按BFS顺序）：
  ├─ 查找所有满足资源需求的SN节点
  ├─ 按noderank降序排序
  ├─ 选择noderank最高的SN节点
  └─ 临时扣减资源
  ↓
恢复临时扣减
  ↓
放置约束节点
  ↓
验证映射并计算路径
  ↓
应用映射（统一扣减资源）
  ↓
返回成功
```

---

## 五、与原始实现的区别

### 原始实现（已修改）

1. **NodeRank计算**：只在初始化时计算一次（基于初始资源）
2. **VN节点排序**：按归一化需求从大到小排序
3. **放置顺序**：按需求排序后的顺序

### 当前实现（符合要求）

1. **NodeRank计算**：每次workflow到达时重新计算（基于剩余资源）✅
2. **起始节点**：从资源消耗最高的VN节点开始 ✅
3. **放置顺序**：按广度优先遍历顺序 ✅
4. **SN节点选择**：优先选择noderank最高的可用SN节点 ✅

---

## 六、算法优势

1. **动态适应**：NodeRank基于剩余资源，适应动态环境
2. **资源优先**：需求大的节点优先放置，减少资源冲突
3. **拓扑感知**：BFS遍历保持VN图的拓扑结构
4. **高效选择**：优先使用资源丰富、位置好的SN节点

---

**文档版本**：v1.0  
**创建时间**：2025年1月

