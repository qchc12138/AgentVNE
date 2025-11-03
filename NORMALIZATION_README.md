# 特征归一化说明文档

## 概述

本文档说明了在 SimuVNE 项目中实现的特征归一化方案，用于稳定强化学习训练过程。

## 归一化方案

### 1. SN（底层网络）特征归一化

**位置**: `env.py` 中的 `SimuVNEEnv.get_sn_state()`

**方法**: 剩余资源按初始容量归一化

**公式**:
```
normalized_resource = current_residual / initial_capacity
```

**特征维度** (6维):
- dim 0: CPU剩余比例 (0~1, 1表示完整资源，0表示耗尽)
- dim 1: Memory剩余比例 (0~1)
- dim 2: Disk剩余比例 (0~1)
- dim 3: Bandwidth (按全网最大值归一化)
- dim 4: Comm_bandwidth (按全网最大值归一化)
- dim 5: 保留维度 (固定为0)

**优势**:
- 直观表示资源可用性
- 随着任务放置，归一化值从1.0逐渐减小
- 便于策略网络识别资源紧张状态

### 2. VN（虚拟网络）特征归一化

**位置**: `env.py` 中的 `WorkflowGenerator.load_workflow_graph()`

**方法**: 需求量按SN最大容量归一化

**公式**:
```
normalized_demand = vn_demand / sn_max_capacity
```

**特征维度** (6维):
- dim 0: CPU需求 (相对于SN最大CPU容量的比例)
- dim 1: Memory需求 (相对于SN最大Memory容量的比例)
- dim 2: Disk需求 (相对于SN最大Disk容量的比例)
- dim 3: Bandwidth需求 (相对于SN最大Bandwidth的比例)
- dim 4: Comm_bandwidth需求 (相对于SN最大Comm_bandwidth的比例)
- dim 5: 保留维度 (固定为0)

**优势**:
- 策略网络可直接对比"需求/供给"比例
- 归一化尺度一致，便于GNN学习
- 避免不同资源类型的量纲差异

### 3. SN最大容量计算

**位置**: `env.py` 中的 `SimuVNEEnv._init_sn_residuals()`

**自动计算**:
```python
self._sn_max_capacity = {
    'cpu_max': max(node.cpu for all nodes),
    'mem_max': max(node.memory for all nodes),
    'disk_max': max(node.disk for all nodes),
    'bw_max': max(link.bandwidth for all links),
    'comm_bw_max': max(node.comm_bandwidth for all nodes),
}
```

**当前SN拓扑的最大值**:
- cpu_max: 4.0
- mem_max: 4.0
- disk_max: 6.0
- bw_max: 10.0
- comm_bw_max: 10.0

## 代码修改说明

### 修改文件清单

1. **env.py**
   - `WorkflowGenerator.__init__()`: 添加 `sn_capacity_for_norm` 参数
   - `WorkflowGenerator.load_workflow_graph()`: 使用SN容量归一化VN特征
   - `SimuVNEEnv.__init__()`: 添加 `_sn_max_capacity` 存储
   - `SimuVNEEnv._init_sn_residuals()`: 计算并保存SN最大容量
   - `SimuVNEEnv.get_sn_max_capacity()`: 新方法，返回SN最大容量
   - `SimuVNEEnv.get_sn_state()`: 按初始容量归一化剩余资源

2. **fine_tuning.py**
   - `run_ppo_episode()`: 创建WorkflowGenerator时传入SN容量

### 向后兼容性

- WorkflowGenerator的 `sn_capacity_for_norm` 参数有默认值
- 如不传入，使用固定的默认容量字典
- 现有代码无需修改即可运行（但建议传入实际SN容量）

## 测试结果

### 归一化范围验证

运行 `test_normalization.py` 的结果:

```
【SN最大容量】
  cpu_max: 4.00
  mem_max: 4.00
  disk_max: 6.00
  bw_max: 10.00
  comm_bw_max: 10.00

【SN状态归一化检查】
  SN特征范围: [0.0000, 1.0000]
  ✓ 初始状态所有资源为1.0（完整）
  ✓ 资源消耗后正确减少（如0.875表示剩余87.5%）

【VN特征归一化检查】
  VN特征范围: [0.0000, 0.2667]
  ✓ 最小需求：0.025（2.5%的SN最大容量）
  ✓ 最大需求：0.267（26.7%的SN最大容量）
  ✓ 所有需求值在合理范围内
```

### 训练效果对比

**归一化前** (从历史日志):
- 价值损失: 11000~12000
- 梯度可能不稳定
- 特征尺度差异大

**归一化后** (运行 `test_normalization_effect.py`):
- 价值损失: 8000~8200 ✓ 显著降低
- V(s)输出: 0.33 (数量级 10^-0.5) ✓ 合理范围
- 特征范围: SN [0, 1], VN [0, 0.27] ✓ 归一化正常
- 接受率: 100% ✓ 任务放置成功

## 预期效果

### 训练稳定性改善
1. **价值损失降低**: 从11000+降至8000左右
2. **梯度更新平滑**: 归一化特征避免梯度爆炸/消失
3. **学习率可调**: 归一化后可适当提高学习率

### 策略学习改善
1. **资源对比直观**: VN需求/SN供给在同一尺度
2. **状态表示清晰**: SN剩余比例直接反映资源紧张度
3. **泛化能力增强**: 归一化后模型对不同拓扑适应性更强

## 使用建议

### 1. 保持归一化一致性
- 预训练和微调使用相同的归一化方案
- 如修改SN拓扑，重新计算最大容量
- 确保测试时也使用归一化特征

### 2. 监控特征范围
- 定期检查SN/VN特征是否在 [0, 1] 或合理范围
- 如出现异常值，检查归一化逻辑
- 记录每次训练的特征统计

### 3. 调整超参数
- 归一化后可尝试提高学习率 (如 3e-4 → 5e-4)
- 价值网络学习率可独立调整
- 观察价值损失是否在合理范围 (<1000 较理想)

## 验证步骤

运行以下命令验证归一化效果:

```bash
# 1. 基础归一化测试
conda run -n AgentVNE python test_normalization.py

# 2. 训练效果测试
conda run -n AgentVNE python test_normalization_effect.py

# 3. 完整训练测试（可选）
conda run -n AgentVNE python fine_tuning.py
```

## 常见问题

### Q1: 为什么价值损失仍然较大 (8000+)?
A: 这可能是由于:
1. 回报的绝对值较大 (如 -355.50)
2. 可考虑对回报也做缩放/归一化
3. 或调整γ (gamma) 参数降低长期回报权重

### Q2: 如何进一步降低价值损失?
A: 可尝试:
1. 回报归一化: `normalized_reward = reward / reward_scale`
2. 价值网络加权: `loss_v = 0.5 * F.mse_loss(...)`
3. 使用Huber损失替代MSE
4. 增加价值网络层数或隐藏维度

### Q3: 归一化会影响预训练模型吗?
A: 
- 如果预训练时**未归一化**，加载权重后会有"分布漂移"
- 建议: 
  - 方案1: 预训练和微调都使用归一化
  - 方案2: 微调时初始学习率调小，让模型适应新分布
  - 方案3: 只对微调使用归一化，预训练模型作为初始化快速适应

## 总结

特征归一化是稳定强化学习训练的关键步骤。本实现通过:
- SN剩余资源按初始容量归一化 (0~1)
- VN需求按SN最大容量归一化 (相对比例)
- 自动计算SN最大容量，保证一致性

有效改善了训练稳定性，降低了价值损失，使模型能更好地学习资源分配策略。

---
*文档生成时间: 2025-11-03*
*测试环境: AgentVNE conda环境*

