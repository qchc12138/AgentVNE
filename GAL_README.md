# GAL (Greedy Allocation Algorithm) - 对比基准算法

## 算法概述

GAL是一个贪心放置算法，作为神经网络方法（PPO + SimuVNE）的对比基准。

### 核心思想

使用简单的贪心策略进行VN到SN的节点映射：
1. 按需求强度排序VN节点（从大到小）
2. 按剩余资源排序SN节点（从大到小）
3. 贪心匹配：优先将高需求VN节点放置到高资源SN节点

## 算法流程

### 1. VN节点排序

```python
# 对每个VN节点，计算归一化需求强度
demand_strength = (cpu_norm + mem_norm + disk_norm)

# 按需求强度降序排序
vn_nodes_sorted = sorted(vn_nodes, key=lambda x: x.demand_strength, reverse=True)
```

### 2. SN节点排序（动态）

```python
# 对每个SN节点，计算当前剩余资源强度（归一化）
resource_strength = (cpu_res_norm + mem_res_norm + disk_res_norm)

# 按剩余资源降序排序
sn_nodes_sorted = sorted(sn_nodes, key=lambda x: x.resource_strength, reverse=True)
```

### 3. 贪心匹配

```python
for vn_node in vn_nodes_sorted:
    # 找到第一个能容纳该VN节点的SN节点（资源最多）
    for sn_node in sn_nodes_sorted:
        if sn_node.can_fit(vn_node):
            mapping[vn_node] = sn_node
            break
    
    # 如果没有合适的SN节点，放置失败
    if vn_node not in mapping:
        return False
```

## 与神经网络方法的对比

| 维度 | GAL (贪心) | PPO + SimuVNE (神经网络) |
|------|-----------|------------------------|
| **决策依据** | 局部资源状态 | 全局图结构 + 历史经验 |
| **优化目标** | 贪心匹配 | 长期回报最大化 |
| **学习能力** | 无（固定策略） | 有（通过训练改进） |
| **计算复杂度** | O(N_v × N_s) | O(GNN forward) |
| **需要训练** | 否 | 是 |
| **适应性** | 固定规则 | 可适应不同场景 |

## 与环境的一致性

GAL与PPO训练使用**完全相同**的环境设置：

### 相同的组件
- ✅ `SimuVNEEnv`: 环境管理
- ✅ `WorkflowGenerator`: 任务生成（泊松到达、指数生存时间）
- ✅ 特征归一化：VN和SN特征归一化策略
- ✅ 资源管理：扣减、恢复逻辑
- ✅ 奖励计算：`_compute_rt()` 方法
- ✅ 最终回报：`compute_final_return()` 方法

### 唯一的区别
**放置策略**：
- **GAL**: 贪心算法（本文件实现）
- **PPO**: 神经网络策略（fine_tuning.py）

## 代码结构

### 核心类

#### `GreedyAllocator`
```python
class GreedyAllocator:
    def __init__(self, env: SimuVNEEnv)
    def greedy_place(self, vn: Data) -> (bool, Dict[int, int], float)
```

**功能**：
- 实现贪心放置逻辑
- 输入：VN图（归一化特征）
- 输出：(成功标志, 节点映射, 奖励)

### 核心函数

#### `run_gal_episode()`
运行单个episode，与`fine_tuning.py`的`run_ppo_episode()`结构完全一致：
- 时间驱动循环
- 泊松任务到达
- 贪心放置决策
- 资源管理
- 轨迹记录

#### `run_gal_benchmark()`
运行多个episode进行统计测试：
- 批量运行N个episode
- 统计平均性能
- 保存结果到JSON

## 使用方法

### 快速测试（单个episode）

```python
from GAL import run_gal_episode

result = run_gal_episode(
    sn_topology_path='/home/zrz/SimuVNE/topo/SN_topology.json',
    workflow_types={'workflow1': '/home/zrz/SimuVNE/workflow_topo/workflow1_topo.json'},
    arrival_rate=0.8,
    mean_lifetime=10.0,
    max_arrived_tasks=30,
    episode_seed=42,
    verbose=True
)

print(f"接受率: {result['acceptance_rate']:.2%}")
print(f"最终回报: {result['final_return']:.2f}")
```

### 基准测试（多个episode）

```python
from GAL import run_gal_benchmark

summary = run_gal_benchmark(
    sn_topology_path='/home/zrz/SimuVNE/topo/SN_topology.json',
    workflow_types={'workflow1': '/home/zrz/SimuVNE/workflow_topo/workflow1_topo.json'},
    arrival_rate=0.8,
    mean_lifetime=10.0,
    max_arrived_tasks=30,
    num_episodes=10,  # 运行10个episode
)

print(f"平均接受率: {summary['summary']['avg_acceptance_rate']:.2%}")
print(f"平均回报: {summary['summary']['avg_return']:.2f}")
```

### 命令行运行

```bash
cd /home/zrz/SimuVNE
conda run -n AgentVNE python GAL.py
```

## 输出结果

### 控制台输出
```
================================================================================
GAL (Greedy Allocation Algorithm) - 基准测试
================================================================================
配置:
  到达率: 0.8
  平均生存时间: 10.0
  每个episode最大任务数: 30
  测试episode数: 10
================================================================================

【Episode 1/10】
    [t=2.0] 任务 #0 到达 (类型:workflow1, 节点数:6, 生存时间:1.7) → ✓成功 (r_t=-0.000, 存活任务数:1)
    ...
    Episode完成: 时间步=50, 到达=30, 接受=28, 接受率=93.3%, 最终回报=-120.50

...

================================================================================
测试结果统计
================================================================================
平均最终回报: -130.25
平均接受率: 91.50%
平均到达任务数: 30.0
平均接受任务数: 27.5
总耗时: 5.20秒
```

### JSON结果文件
保存在 `gal_outputs/gal_results_YYYYMMDD_HHMMSS.json`

```json
{
  "timestamp": "20251103_123456",
  "config": {
    "arrival_rate": 0.8,
    "mean_lifetime": 10.0,
    "max_arrived_tasks": 30,
    "num_episodes": 10
  },
  "summary": {
    "avg_return": -130.25,
    "avg_acceptance_rate": 0.915,
    "avg_arrived": 30.0,
    "avg_accepted": 27.5,
    "total_time": 5.20
  },
  "results": [
    {
      "final_return": -120.50,
      "arrived": 30,
      "accepted": 28,
      "acceptance_rate": 0.933,
      ...
    },
    ...
  ]
}
```

## 性能评估指标

### 主要指标
1. **接受率 (Acceptance Rate)**
   - 公式：`accepted / arrived`
   - 越高越好

2. **最终回报 (Final Return)**
   - 公式：`T_total / T_p × Σr_t`
   - 考虑了跳数和接受率
   - 越大越好（负值，越接近0越好）

3. **平均跳数**
   - 从轨迹中的 `r_t` 计算
   - 越小越好

### 与PPO对比
运行GAL和PPO在相同配置下的测试，对比：
- 接受率
- 最终回报
- 资源利用效率

## 实现细节

### 资源单位一致性

GAL正确处理了归一化与绝对值的转换：

```python
# VN特征已归一化
norm_cpu = vn.x[i][0].item()  # 归一化需求

# 转为绝对值用于资源检查
abs_cpu = norm_cpu * sn_max_capacity['cpu_max']

# 检查SN剩余资源（绝对值）
if sn_res_cpu >= abs_cpu:
    # 可放置
```

### 贪心策略的优缺点

#### 优点
- ✅ 简单高效
- ✅ 不需要训练
- ✅ 可解释性强
- ✅ 作为基准线（baseline）

#### 缺点
- ❌ 局部最优（可能错过全局最优）
- ❌ 不考虑链路拓扑
- ❌ 无法从经验中学习
- ❌ 对资源碎片化不敏感

## 扩展建议

### 可能的改进方向
1. **考虑链路开销**：优先选择距离近的SN节点
2. **资源平衡**：避免将所有高需求VN节点放到同一SN节点
3. **动态调整**：根据历史接受率调整贪心策略

### 多种贪心变体
- **First-Fit**: 第一个满足需求的SN节点
- **Best-Fit**: 剩余资源最接近需求的SN节点
- **Worst-Fit**: 剩余资源最多的SN节点（当前实现）

## 总结

GAL提供了一个简单但有效的基准线，用于评估神经网络方法的性能增益。通过与PPO+SimuVNE的对比，可以量化学习方法带来的改进。

---

**文件**: `GAL.py`  
**作者**: AgentVNE项目组  
**日期**: 2025年11月3日  
**版本**: 1.0

