# r_t 计算说明

## 一、概述

**r_t** 是测试时每个时间步计算的即时奖励值，用于评估当前系统状态（存活的workflow）的性能。

**核心思想**：
- r_t 反映当前存活的workflow的平均跳数（考虑资源竞争）
- r_t 为**负值**，跳数越少，r_t越大（越接近0）
- 只计算当前存活的workflow，不包括失败的任务

---

## 二、计算方法

### 2.1 计算入口

**代码位置**：`env.py` 第446行，`_compute_rt()` 方法

**调用时机**：
1. **每个时间步**：`tests/test_strategy.py` 第221行
   ```python
   current_r_t = float(env._compute_rt())
   time_step_r_t.append({
       "time_step": time_step,
       "r_t": current_r_t,
   })
   ```

2. **任务放置成功后**：`tests/test_strategy.py` 第386行
   ```python
   record["r_t"] = float(env._compute_rt())
   ```

### 2.2 计算步骤

#### 步骤1：统计每条SN边被多少个workflow使用

```python
edge_usage_count: Dict[Tuple[int, int], int] = {}

for wf in self.active_workflows:
    vn_paths = wf['paths']
    edges_in_this_wf: set = set()
    
    # 遍历该workflow的所有路径
    for su, sv, path in vn_paths:
        if len(path) <= 1:
            continue  # 跳过自环（同一SN节点）
        
        # 遍历路径上的每条边
        for a, b in zip(path[:-1], path[1:]):
            edge = (min(a, b), max(a, b))  # 标准化边的表示（无向图）
            edges_in_this_wf.add(edge)
    
    # 统计这个workflow使用的所有边
    for edge in edges_in_this_wf:
        edge_usage_count[edge] = edge_usage_count.get(edge, 0) + 1
```

**说明**：
- `edge_usage_count[edge]` 表示该SN边被多少个workflow使用
- 例如：如果边 `(1, 2)` 被3个workflow使用，则 `edge_usage_count[(1, 2)] = 3`

#### 步骤2：计算每个workflow的加权跳数

```python
total_hops = 0.0

for wf in self.active_workflows:
    vn_paths = wf['paths']
    workflow_hops = 0.0
    
    for su, sv, path in vn_paths:
        if len(path) <= 1:
            continue  # 跳过自环
        
        # 遍历路径上的每条边
        for a, b in zip(path[:-1], path[1:]):
            edge = (min(a, b), max(a, b))
            k = edge_usage_count.get(edge, 1)  # 该边被k个workflow共享
            workflow_hops += k  # 该边贡献k跳
    
    total_hops += workflow_hops
```

**关键点**：
- **加权跳数**：如果一条边被k个workflow共享，则该边贡献k跳
- 这反映了资源竞争的影响：共享的边越多，跳数越大

**示例**：
- 假设workflow A使用路径 `[1, 2, 3]`，边 `(1, 2)` 被3个workflow共享，边 `(2, 3)` 被2个workflow共享
- workflow A的跳数 = 3 + 2 = 5跳

#### 步骤3：计算平均值并取负值

```python
num_workflows = len(self.active_workflows)
if num_workflows == 0:
    return 0.0

avg_hops = total_hops / float(num_workflows)
r_t = -avg_hops
return r_t
```

**说明**：
- `avg_hops`：所有workflow的平均加权跳数
- `r_t = -avg_hops`：取负值，使得跳数越少，r_t越大（越接近0）

---

## 三、计算示例

### 3.1 简单示例

**场景**：
- 有2个存活的workflow
- workflow A：路径 `[1, 2, 3]`，边 `(1, 2)` 被2个workflow共享，边 `(2, 3)` 被1个workflow共享
- workflow B：路径 `[2, 3, 4]`，边 `(2, 3)` 被2个workflow共享，边 `(3, 4)` 被1个workflow共享

**步骤1：统计边使用次数**
```
edge_usage_count = {
    (1, 2): 1,  # 只有workflow A使用
    (2, 3): 2,  # workflow A和B都使用
    (3, 4): 1,  # 只有workflow B使用
}
```

**步骤2：计算每个workflow的跳数**
```
workflow A:
  - 边 (1, 2): k=1 → 贡献1跳
  - 边 (2, 3): k=2 → 贡献2跳
  - workflow A总跳数 = 1 + 2 = 3

workflow B:
  - 边 (2, 3): k=2 → 贡献2跳
  - 边 (3, 4): k=1 → 贡献1跳
  - workflow B总跳数 = 2 + 1 = 3

total_hops = 3 + 3 = 6
```

**步骤3：计算r_t**
```
avg_hops = 6 / 2 = 3.0
r_t = -3.0
```

### 3.2 复杂示例（考虑资源竞争）

**场景**：
- 有3个存活的workflow
- workflow A：路径 `[1, 2, 3]`
- workflow B：路径 `[2, 3, 4]`
- workflow C：路径 `[1, 2, 3, 4]`

**步骤1：统计边使用次数**
```
edge_usage_count = {
    (1, 2): 2,  # workflow A和C使用
    (2, 3): 3,  # workflow A、B、C都使用
    (3, 4): 2,  # workflow B和C使用
}
```

**步骤2：计算每个workflow的跳数**
```
workflow A: 2 + 3 = 5跳
workflow B: 3 + 2 = 5跳
workflow C: 2 + 3 + 2 = 7跳

total_hops = 5 + 5 + 7 = 17
```

**步骤3：计算r_t**
```
avg_hops = 17 / 3 ≈ 5.67
r_t = -5.67
```

---

## 四、关键特性

### 4.1 只计算存活的workflow

**代码**：
```python
num_workflows = len(self.active_workflows)
if num_workflows == 0:
    return 0.0
```

**说明**：
- 只计算 `active_workflows` 中的workflow
- 失败的任务不会被添加到 `active_workflows`，所以不会影响r_t

### 4.2 考虑资源竞争

**加权跳数**：
- 如果一条边被k个workflow共享，则该边贡献k跳
- 这反映了资源竞争的影响：共享的边越多，跳数越大

**示例**：
- 如果边 `(1, 2)` 被3个workflow共享，则该边贡献3跳
- 如果边 `(1, 2)` 只被1个workflow使用，则该边贡献1跳

### 4.3 r_t为负值

**设计原因**：
- r_t = -avg_hops，使得跳数越少，r_t越大（越接近0）
- 这符合奖励的定义：性能越好，奖励越大

**示例**：
- 如果 avg_hops = 5.0，则 r_t = -5.0
- 如果 avg_hops = 3.0，则 r_t = -3.0（更好）

### 4.4 自环处理

**代码**：
```python
if len(path) <= 1:
    continue  # 跳过自环（同一SN节点）
```

**说明**：
- 如果两个VN节点映射到同一个SN节点，路径长度为1，跳数为0
- 这种情况会被跳过，不计算跳数

---

## 五、测试时的使用

### 5.1 每个时间步记录

**代码位置**：`tests/test_strategy.py` 第218-225行

```python
while time_step < cfg.max_time_steps:
    env.step_time(1.0)
    
    # 每个时间步都记录r_t值
    current_r_t = float(env._compute_rt())
    time_step_r_t.append({
        "time_step": time_step,
        "r_t": current_r_t,
    })
```

**说明**：
- 每个时间步都会计算并记录r_t
- 用于绘制 `|r_t|` vs `time_step` 图表

### 5.2 任务放置成功后记录

**代码位置**：`tests/test_strategy.py` 第386行

```python
if record["success"]:
    # ...
    record["r_t"] = float(env._compute_rt())
```

**说明**：
- 任务放置成功后，记录当前的r_t值
- 用于任务级别的r_t记录

---

## 六、与其他指标的关系

### 6.1 avg_hops（平均跳数）

**关系**：
- `avg_hops = -r_t`
- `r_t = -avg_hops`

**说明**：
- avg_hops是正数，表示平均跳数
- r_t是负数，表示奖励值（跳数越少，r_t越大）

### 6.2 与接受率的关系

**独立指标**：
- r_t只反映当前存活的workflow的性能
- 接受率反映任务放置的成功率
- 两者是独立的指标

**示例**：
- 如果接受率低（很多任务失败），但存活的workflow跳数少，r_t仍然可能较大
- 如果接受率高（很多任务成功），但存活的workflow跳数多，r_t可能较小

---

## 七、总结

### 7.1 计算公式

```
r_t = -avg_hops

其中：
avg_hops = total_hops / num_workflows

total_hops = sum(workflow_hops for each workflow)

workflow_hops = sum(k for each edge in path)
k = edge_usage_count[edge]  # 该边被k个workflow共享
```

### 7.2 关键特性

1. **只计算存活的workflow**：失败的任务不影响r_t
2. **考虑资源竞争**：共享的边贡献更多跳数
3. **r_t为负值**：跳数越少，r_t越大（越接近0）
4. **自环处理**：同一SN节点的映射跳数为0

### 7.3 使用场景

- **每个时间步**：记录r_t值，用于绘制时间序列图
- **任务放置成功后**：记录r_t值，用于任务级别的分析
- **性能评估**：r_t反映当前系统状态（存活的workflow）的性能

---

**文档版本**：v1.0  
**创建时间**：2025年1月

