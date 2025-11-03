# 预训练数据归一化说明

## 修改概览

为了与 fine-tuning 保持一致，预训练数据也进行了特征归一化。

## 修改内容

### 1. `dataset_generate.py` 

#### 新增函数：`_compute_sn_max_capacity()`
```python
def _compute_sn_max_capacity(nodes: List[Dict]) -> Dict[str, float]:
    """计算SN网络的最大容量（用于归一化）"""
```

- 遍历所有SN节点，找到每种资源的最大值
- 返回字典：`{'cpu_max', 'mem_max', 'disk_max', 'bw_max', 'comm_bw_max'}`

#### 修改函数：`_nodes_to_features()`
```python
def _nodes_to_features(nodes: List[Dict], 
                       is_workflow: bool = False, 
                       sn_max_capacity: Dict[str, float] = None) -> torch.Tensor:
```

**关键变化：**
- 特征顺序改为：`[cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]`
  - 原来是：`[type, memory, cpu, disk, comm_bw, num_nodes]`
- 所有特征归一化：除以 `sn_max_capacity` 中的对应最大值
- 移除了 `type` 和 `num_nodes` 特征

#### 修改函数：`_topology_to_pyg_data()`
```python
def _topology_to_pyg_data(topo: Dict, 
                          is_workflow: bool = False, 
                          sn_max_capacity: Dict[str, float] = None) -> Data:
```

- 接受 `sn_max_capacity` 参数并传递给 `_nodes_to_features()`

#### 修改函数：`generate_pretrain_dataset()`

- 开始时计算并打印 `sn_max_capacity`
- 生成样本时传递 `sn_max_capacity` 给 `_topology_to_pyg_data()`
- 在保存的 `dataset_info` 中添加：
  - `'sn_max_capacity'`: 归一化参数
  - `'normalized': True`: 标记数据已归一化

### 2. `pretrain.py`

#### 修改函数：`load_pretrain_dataset()`

- 加载数据集后，打印归一化信息
- 显示 `sn_max_capacity` 详细值

## 归一化策略

### 特征映射

| 维度 | 特征 | 归一化方式 |
|------|------|-----------|
| 0 | CPU | `cpu / cpu_max` |
| 1 | Memory | `memory / mem_max` |
| 2 | Disk | `disk / disk_max` |
| 3 | Bandwidth | `bandwidth / bw_max` |
| 4 | Comm Bandwidth | `comm_bandwidth / comm_bw_max` |
| 5 | Padding | 固定为 0.0 |

### 归一化参数（默认）

基于 `/home/zrz/SimuVNE/topo/SN_topology.json`：

```python
{
    'cpu_max': 4.0,
    'mem_max': 4.0,
    'disk_max': 6.0,
    'bw_max': 10.0,
    'comm_bw_max': 10.0
}
```

### VN 和 SN 使用相同的归一化

- **VN（需求）** 和 **SN（容量）** 都除以 SN 的最大容量
- 这样可以直接比较 VN 需求是否能被 SN 容量满足

## 与 fine-tuning 的一致性

### `env.py` (fine-tuning)
```python
# WorkflowGenerator.load_workflow_graph
x_list.append([
    float(n.get('cpu', 0.0)) / (self.sn_capacity['cpu_max'] + 1e-8),
    float(n.get('memory', 0.0)) / (self.sn_capacity['mem_max'] + 1e-8),
    float(n.get('disk', 0.0)) / (self.sn_capacity['disk_max'] + 1e-8),
    ...
])

# SimuVNEEnv.get_sn_state
x_list.append([
    nd['cpu_res'] / (init['cpu'] + 1e-8),
    nd['mem_res'] / (init['mem'] + 1e-8),
    nd['disk_res'] / (init['disk'] + 1e-8),
    ...
])
```

### `dataset_generate.py` (预训练)
```python
# _nodes_to_features
feats.append([
    cpu / (sn_max_capacity['cpu_max'] + 1e-8),
    memory / (sn_max_capacity['mem_max'] + 1e-8),
    disk / (sn_max_capacity['disk_max'] + 1e-8),
    ...
])
```

**特征顺序一致：** `[cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]`

## 重新生成数据集

旧数据集未归一化，需要重新生成：

```bash
cd /home/zrz/SimuVNE
conda run -n AgentVNE python dataset_generate.py
```

## 验证归一化

测试脚本：`test_pretrain_normalization.py`

```bash
conda run -n AgentVNE python test_pretrain_normalization.py
```

**预期输出：**
- SN 特征范围：`[0.0, 1.0]`
- VN 特征范围：`[0.0, 1.0]`
- 数据集标记为 `normalized: True`

## 影响

### 优点
1. **训练稳定性提升**：输入范围统一，减少数值问题
2. **与fine-tuning一致**：预训练模型可直接用于fine-tuning
3. **模型泛化性**：不依赖具体资源值的尺度

### 注意事项
1. **旧模型不兼容**：用旧数据训练的模型需要重新训练
2. **资源检查要反归一化**：在 `env.py` 中已处理（见 `_check_node_feasible` 等函数）

## 总结

预训练和fine-tuning现在使用完全一致的特征归一化策略，确保了端到端的训练流程统一性。

