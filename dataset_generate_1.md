# dataset_generate_1.py 详细文档

## 概述

`dataset_generate_1.py` 是一个预训练数据集生成脚本，用于为 VNE (Virtual Network Embedding) 模型生成监督学习训练数据。该脚本基于底层网络（SN, Substrate Network）与工作流（Workflow/VN, Virtual Network）的拓扑结构，通过贪心放置策略生成 `<x, y>` 预训练数据集。

## 核心功能

1. **带偏置的 SN 拓扑生成**：为约束节点添加 `bias_cpu` 和 `bias_bandwidth` 字段
2. **NodeRank 计算**：基于 `(cpu + bias_cpu) * bandwidth` 计算 SN 节点的 NodeRank
3. **贪心/BFS 放置策略**：将 workflow 节点映射到底层网络节点
4. **样本生成**：记录每次放置前的 SN/VN 状态和对应的 NodeRank 标签
5. **数据集保存**：将所有样本保存为 PyTorch 格式文件

## 工作流程

### 1. 初始化阶段

```
输入文件校验
  ↓
读取 Workflow 拓扑，提取 constraint_node
  ↓
调用 SN_topo_addbias.add_bias_to_topology()
  生成 SN_topology_2.json（带 bias_cpu/bias_bandwidth）
  ↓
加载偏置后的 SN 拓扑和 Workflow NodeRank
  ↓
计算 SN 最大容量（用于特征归一化）
```

**关键步骤**：
- 从 workflow 拓扑中提取唯一的 `constraint_node`
- 调用 `add_bias_to_topology()` 为约束节点添加偏置资源
  - 约束节点：`bias_cpu = bias * max_cpu`
  - 其他节点：`bias_cpu = 0`
  - 所有节点：`bias_bandwidth = comm_bandwidth`

### 2. 样本生成循环

对每个 **Episode**（共 `num_episodes` 个）：

```
重置 SN 拓扑（深拷贝，恢复初始资源）
  ↓
对每个 Workflow（共 workflows_per_episode 个）：
  ├─ 计算当前 SN 的 NodeRank（使用扣减后的资源 + 固定 bias_cpu）
  ├─ 构建 Workflow 和 SN 的图特征（PyTorch Geometric Data）
  ├─ 生成标签 y：将 SN NodeRank 重复 N1 行（N1×N2 矩阵）
  ├─ 保存样本：{workflow_graph, substrate_graph, label}
  ├─ 执行贪心/BFS 放置：
  │   ├─ 步骤1：放置约束节点到指定 SN 节点
  │   ├─ 步骤2：选择资源需求最大的非约束 VN 节点
  │   ├─ 步骤3：BFS 扩展放置（优先同 SN，否则 k-hop 邻居）
  │   └─ 步骤4：扣减 SN 资源（仅扣减 cpu/memory/disk，bias_cpu 不变）
  └─ 更新进度条
```

**重要特性**：
- NodeRank 在每次放置前重新计算，使用**当前剩余资源** + **固定 bias_cpu**
- 资源扣减只影响 `cpu`、`memory`、`disk`，`bias_cpu` 保持不变
- 放置失败时会回滚所有资源扣减

### 3. NodeRank 计算详解

`_compute_sn_noderank()` 函数实现：

```python
# 步骤1：资源评估 H(u) = (cpu + bias_cpu) * bandwidth
H[i] = (cpu + bias_cpu) * bandwidth

# 步骤2：初始 NodeRank（归一化）
NR_0 = H / sum(H)

# 步骤3：构建邻接矩阵（调用 calculate_noderank_2）
adj_info = build_adjacency_info(nodes, links, directed)

# 步骤4：计算前向概率矩阵
pF = calculate_forward_probability(H, adjacency)

# 步骤5：迭代传播（2轮）
for _ in range(2):
    NR_next = NR_curr + 0.20 * (pF @ NR_curr)
    NR_curr = NR_next / sum(NR_next)

# 步骤6：三次幂 + 归一化
NR_final = (NR_curr ** 3) / sum(NR_curr ** 3)
```

### 4. 贪心/BFS 放置策略

`_greedy_place_workflow()` 函数实现：

**步骤1：约束节点优先**
- 遍历所有 VN 节点，找到有 `constraint_node` 字段的节点
- 直接放置到指定的 SN 节点
- 检查资源是否足够，不足则回滚并返回失败

**步骤2：生成优先级列表**
- 基于 SN NodeRank 降序排序
- 每个 VN 节点使用相同的优先级列表

**步骤3：选择首个非约束节点**
- 选择资源需求最大的非约束 VN 节点
- 按优先级列表顺序尝试放置

**步骤4：BFS 扩展放置**
- 队列初始化为首个非约束节点
- 对队列中每个节点 `vi`：
  - 找到 `vi` 的未放置邻居 `u`
  - **策略1**：尝试将 `u` 放在与 `vi` 相同的 SN 节点上
  - **策略2**：如果策略1失败，在 k-hop 邻居中按优先级搜索（k 从 1 开始，逐步扩展）
  - 成功放置后，将 `u` 加入队列
- 队列更新：按度降序，度相同则按资源需求降序

**步骤5：资源管理**
- 成功放置时立即扣减资源
- 放置失败时回滚所有资源扣减
- 只扣减 `cpu`、`memory`、`disk`，`bias_cpu` 保持不变

### 5. 图特征构建

**节点特征**（6维）：
- `[cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]`
- 所有特征按 SN 最大容量归一化

**边索引**：
- 使用 `torch_geometric.data.Data` 格式
- 无向图：每条边添加双向连接

### 6. 数据集保存

保存格式（PyTorch `.pt` 文件）：

```python
{
    'samples': [
        {
            'workflow_graph': Data(x=[N1, 6], edge_index=[2, E1]),
            'substrate_graph': Data(x=[N2, 6], edge_index=[2, E2]),
            'label': Tensor([N1, N2])  # SN NodeRank 矩阵
        },
        ...
    ],
    'info': {
        'num_samples': int,
        'workflows_per_episode': int,
        'num_episodes': int,
        'sn_topo_path': str,
        'workflow_topo_path': str,
        'workflow_noderank_path': str,
        'sn_max_capacity': Dict[str, float],
        'normalized': True,
        'bias': float
    }
}
```

## 输入参数

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sn_topo` | str | `/home/zrz/AgentVNE/AgentVNE/topo/SN_topology_2.json` | 带偏置的 SN 拓扑输出路径 |
| `--workflow_topo` | str | `workflow_topo/workflow1_topo.json` | Workflow 拓扑文件路径 |
| `--workflow_noderank` | str | `workflow_topo/workflow1_noderank.json` | Workflow NodeRank 文件路径 |
| `--output` | str | `pretrain_data/pretrain_dataset.pt` | 输出数据集文件路径 |
| `--test_output` | str | `pretrain_data/test_sample.pt` | 测试样本输出文件路径（可选） |
| `--workflows_per_episode` | int | 7 | 每个 episode 放置的 workflow 数量 |
| `--num_episodes` | int | 400 | Episode 数量（重复次数） |
| `--test_mode` | flag | False | 启用测试模式，打印放置详情 |
| `--test_episode_idx` | int | 0 | 测试模式下要打印的 episode 索引 |
| `--bias` | float | 0.5 | bias 参数（约束节点 `bias_cpu = bias * max_cpu`） |

### 输入文件格式

#### SN 拓扑文件（JSON）

```json
{
    "nodes": [
        {
            "id": 0,
            "cpu": 4.0,
            "memory": 4.0,
            "disk": 6.0,
            "bandwidth": 10.0,
            "comm_bandwidth": 10.0
        },
        ...
    ],
    "links": [
        {
            "source": 0,
            "target": 1,
            "bandwidth": 10.0
        },
        ...
    ],
    "directed": false
}
```

#### Workflow 拓扑文件（JSON）

```json
{
    "nodes": [
        {
            "id": 0,
            "cpu": 1.0,
            "memory": 1.0,
            "disk": 1.0,
            "constraint_node": 6  // 可选，指定必须放置的 SN 节点 ID
        },
        ...
    ],
    "links": [
        {
            "source": 0,
            "target": 1,
            "bandwidth": 1.0
        },
        ...
    ],
    "directed": false
}
```

#### Workflow NodeRank 文件（JSON）

```json
{
    "noderank": [0.1, 0.2, 0.15, ...]  // 每个 VN 节点的 NodeRank 值
}
```

## 输出文件

### 1. 预训练数据集文件（`.pt`）

**路径**：`pretrain_data/pretrain_dataset.pt`

**内容**：
- `samples`: 样本列表，每个样本包含：
  - `workflow_graph`: PyTorch Geometric Data 对象（VN 图）
  - `substrate_graph`: PyTorch Geometric Data 对象（SN 图）
  - `label`: `[N1, N2]` 张量，每行是 SN NodeRank 向量
- `info`: 数据集元信息

**样本数量**：`num_episodes × workflows_per_episode`

### 2. 测试样本文件（可选，`.pt`）

**路径**：`pretrain_data/test_sample.pt`

**内容**：单条测试样本，格式与数据集相同

### 3. SN_topology_2.json

**路径**：`/home/zrz/AgentVNE/AgentVNE/topo/SN_topology_2.json`

**内容**：带偏置的 SN 拓扑，每个节点包含：
- 原始资源字段：`cpu`, `memory`, `disk`, `bandwidth`, `comm_bandwidth`
- 新增偏置字段：`bias_cpu`, `bias_bandwidth`（或 `nr_cpu`, `nr_bandwidth`）

## 关键函数说明

### `_compute_sn_noderank()`

计算 SN 节点的 NodeRank，使用公式：
- 资源评估：`H(u) = (cpu + bias_cpu) * bandwidth`
- 迭代传播：2 轮前向传播 + 三次幂归一化

### `_greedy_place_workflow()`

贪心/BFS 放置策略：
1. 约束节点优先放置
2. 选择资源需求最大的非约束节点作为起点
3. BFS 扩展：优先同 SN，否则 k-hop 邻居
4. 按 NodeRank 优先级排序

### `_topology_to_pyg_data()`

将拓扑字典转换为 PyTorch Geometric Data 对象：
- 节点特征：6 维归一化特征
- 边索引：`[2, E]` 格式

### `_nodes_to_features()`

节点特征提取和归一化：
- 特征顺序：`[cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]`
- 归一化：除以 SN 最大容量

## 使用示例

### 基本用法

```bash
python3 dataset_generate_1.py
```

### 自定义参数

```bash
python3 dataset_generate_1.py \
    --workflows_per_episode 10 \
    --num_episodes 50 \
    --bias 0.4 \
    --output /path/to/output.pt \
    --test_mode \
    --test_episode_idx 0
```

### 测试模式

```bash
python3 dataset_generate_1.py --test_mode --test_episode_idx 0
```

测试模式会打印：
- 放置前每个 SN 节点的 NodeRank
- 每个 workflow 的放置步骤
- 资源变化情况
- 放置失败详情（如果有）

## 注意事项

1. **bias_cpu 不参与资源扣减**：`bias_cpu` 仅用于 NodeRank 计算，不影响实际资源可用性
2. **NodeRank 动态更新**：每次放置前都会重新计算 NodeRank，使用当前剩余资源
3. **约束节点唯一性**：每个 workflow 只能有一个 `constraint_node`
4. **资源回滚机制**：放置失败时会回滚所有已扣减的资源
5. **特征归一化**：所有节点特征都按 SN 最大容量归一化
6. **文件路径**：支持绝对路径和相对路径（相对路径相对于脚本目录）

## 依赖模块

- `torch`: PyTorch 深度学习框架
- `torch_geometric`: 图神经网络库
- `networkx`: 图算法库
- `numpy`: 数值计算库
- `tqdm`: 进度条显示
- `topo.calculate_noderank_2`: NodeRank 计算模块
- `topo.SN_topo_addbias`: SN 拓扑偏置生成模块

## 输出示例

### 控制台输出

```
================================================
开始生成预训练数据集
SN 输出文件: /home/zrz/AgentVNE/AgentVNE/topo/SN_topology_2.json
workflow: /home/zrz/AgentVNE/AgentVNE/workflow_topo/workflow1_topo.json
bias: 0.4
7 workflows / episode × 400 episodes
================================================

校验输入文件...
✓ 原始 SN 拓扑: /home/zrz/AgentVNE/AgentVNE/topo/SN_topology.json
✓ Workflow 拓扑: /home/zrz/AgentVNE/AgentVNE/workflow_topo/workflow1_topo.json
✓ Workflow NodeRank: /home/zrz/AgentVNE/AgentVNE/workflow_topo/workflow1_noderank.json

constraint_node: 6

生成带偏置的 SN_topology_2.json ...

开始生成 2800 个样本...
生成样本: 100%|████████████| 2800/2800 [00:30<00:00, 92.5it/s]

=== 最后一个 Episode 400/400 ===

Episode 400, Workflow 1 放置前 NodeRank：
  SN节点 0 (ID=0): 0.123456
  SN节点 1 (ID=1): 0.234567
  ...

Episode 400, Workflow 1: 放置 7/7
  步骤1: VN 0 -> SN 6 (前: cpu=4.00, 后: cpu=3.00)
  步骤2: VN 1 -> SN 6 (前: cpu=3.00, 后: cpu=2.00)
  ...

保存数据集到 /home/zrz/AgentVNE/AgentVNE/pretrain_data/pretrain_dataset.pt ...
================================================
完成，样本数: 2800，文件大小: 15.23 MB
================================================
```

## 总结

`dataset_generate_1.py` 是一个完整的数据集生成工具，它：

1. **生成带偏置的 SN 拓扑**：为约束节点添加偏置资源
2. **动态计算 NodeRank**：基于当前资源状态和偏置值
3. **执行贪心放置**：使用 BFS 扩展策略进行 VN 映射
4. **生成训练样本**：记录图状态和对应的 NodeRank 标签
5. **保存数据集**：以 PyTorch 格式保存，便于后续训练

该脚本生成的数据集可用于预训练 VNE 模型，学习如何根据 SN 和 VN 的状态预测最优的节点映射方案。

