阅读文件，梳理工作流程：


[3 tools called]


查看核心函数的工作流程：


[3 tools called]


## `dataset_generate.py` 工作流程介绍

### 一、总体目标
生成用于预训练的 `<x, y>` 数据集，其中：
- `x`：workflow 拓扑与当前 SN（底层网络）状态
- `y`：基于 NodeRank 的分配概率矩阵（N1×N2，N1 为 VN 节点数，N2 为 SN 节点数）

### 二、核心工作流程

#### 阶段1：初始化与数据加载
1. 加载输入文件：
   - SN 拓扑（`sn_topo_path`）
   - Workflow 拓扑（`workflow_topo_path`）
   - Workflow NodeRank（`workflow_noderank_path`）
2. 计算 SN 最大容量：
   - 用于归一化节点特征
   - 包括 CPU、Memory、Disk、Bandwidth、Comm_Bandwidth

#### 阶段2：生成测试样本（可选）
- 生成一条初始状态的测试样本（未放置任何 workflow）
- 包含初始 SN 与 workflow 的图结构

#### 阶段3：主循环生成训练样本

外层循环：Episode（默认 50 次）
- 每次 episode 重置 SN 到初始状态（深拷贝）

内层循环：Workflow 放置（每个 episode 放置多个 workflow，默认 5 个）

对每个 workflow，执行以下步骤：

步骤1：提取约束节点信息
- 检查 workflow 中 VN 节点的 `constraint_node` 字段
- 收集需要添加偏置的 SN 节点 ID

步骤2：计算 SN NodeRank（考虑约束节点偏置）
```python
_compute_sn_noderank() 函数：
1. 如果有约束节点，为对应SN节点添加偏置资源（仅用于计算）
2. 计算资源评估：H(u) = CPU × comm_bandwidth
3. 计算初始NodeRank：NR_0 = H / ΣH
4. 构建邻接矩阵和前向概率矩阵
5. 进行2轮邻居传播（PF_u=0.20）
6. 对结果做3次幂并归一化
```

步骤3：构建输入特征 x
- Workflow 图：转换为 PyTorch Geometric 的 Data 对象
- SN 图：转换为 PyTorch Geometric 的 Data 对象
  - 如果有约束节点，使用偏置后的资源特征
  - 节点特征归一化：[cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]

步骤4：生成标签 y
- 将 SN 的 NodeRank 值重复 N1 行，形成 N1×N2 矩阵
- 每行表示一个 VN 节点对所有 SN 节点的分配概率

步骤5：保存样本
- 将 `(workflow_graph, substrate_graph, label)` 添加到样本列表

步骤6：贪心放置 workflow
```python
_greedy_place_workflow() 函数：
1. 先处理约束节点：
   - 直接放置到 constraint_node 指定的 SN 节点
   - 检查资源是否足够
   - 约束节点不入队列，不参与BFS扩展

2. 选择第一个非约束节点：
   - 选择资源需求最大的VN节点
   - 按SN NodeRank降序尝试放置

3. BFS扩展放置：
   - 从已放置节点开始BFS扩展
   - 优先放在同一SN节点上
   - 资源不足时在k跳邻居中查找
   - 按NodeRank优先级排序

4. 资源管理：
   - 成功放置后扣减SN资源
   - 失败时回滚所有资源扣减
```

步骤7：更新 SN 资源
- 根据放置结果扣减 SN 节点的 CPU、Memory、Disk
- 为下一个 workflow 的放置做准备

### 三、关键特性

#### 1. 约束节点支持
- 支持 VN 节点的 `constraint_node` 字段
- 约束节点直接放置到指定 SN 节点
- 计算 NodeRank 时为约束节点对应的 SN 节点添加偏置

#### 2. 偏置机制
- 偏置因子（`bias_factor`，默认 0.5）可通过命令行参数调整
- 偏置仅用于 NodeRank 计算和特征生成，不影响实际资源

#### 3. 数据归一化
- 所有节点特征基于 SN 最大容量归一化
- 确保不同规模的网络数据可统一处理

#### 4. 贪心放置策略
- BFS 扩展：优先将相邻 VN 节点放在同一 SN 节点
- NodeRank 优先级：按 NodeRank 值排序选择 SN 节点
- 资源约束检查：确保 CPU、Memory、Disk 满足需求

### 四、输出结果

最终生成的数据集包含：
- `samples`：样本列表，每个样本包含：
  - `workflow_graph`：PyTorch Geometric Data 对象
  - `substrate_graph`：PyTorch Geometric Data 对象
  - `label`：NodeRank 分配概率矩阵（N1×N2）
- `info`：数据集元信息：
  - 样本数量、episode 数量
  - 文件路径、归一化参数等

### 五、使用示例

```bash
python dataset_generate.py \
    --sn_topo /path/to/SN_topology.json \
    --workflow_topo /path/to/workflow1_topo.json \
    --workflow_noderank /path/to/workflow1_noderank.json \
    --output /path/to/pretrain_dataset.pt \
    --workflows_per_episode 5 \
    --num_episodes 400 \
    --bias_factor 0.5 \
    --test_mode
```

该脚本用于生成虚拟网络嵌入（VNE）的预训练数据集，通过模拟多个 workflow 的放置过程，学习基于 NodeRank 的节点分配策略。