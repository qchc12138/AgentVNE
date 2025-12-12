# r_t 惩罚机制分析

## 一、问题：放置失败时是否对r_t进行惩罚？

## 二、代码分析

### 2.1 测试时的r_t计算

**代码位置**：`tests/test_strategy.py` 和 `env.py`

**流程**：
1. 任务到达时，调用 `strategy.place()` 进行放置
2. 如果放置成功（`result.success == True`），任务被添加到 `env.active_workflows`
3. 如果放置失败（`result.success == False`），任务**不会被添加**到 `env.active_workflows`

**r_t计算**：`env._compute_rt()`（`env.py` 第446行）
```python
def _compute_rt(self) -> float:
    num_workflows = len(self.active_workflows)
    if num_workflows == 0:
        return 0.0
    # ... 只计算 active_workflows 中的workflow的跳数
    # r_t = -avg_hops
    return r_t
```

**关键点**：
- `_compute_rt()` **只计算** `active_workflows` 中的workflow
- 失败的任务**不在** `active_workflows` 中
- 所以**失败的任务不会影响r_t的计算**

### 2.2 每个时间步的r_t记录

**代码位置**：`tests/test_strategy.py` 第218-225行

```python
while time_step < cfg.max_time_steps:
    env.step_time(1.0)
    # 每个时间步都记录r_t值
    current_r_t = float(env._compute_rt())  # 只计算存活的workflow
    time_step_r_t.append({
        "time_step": time_step,
        "r_t": current_r_t,
    })
```

**结论**：每个时间步记录的r_t只包括当前存活的workflow，不包括失败的任务。

### 2.3 放置失败时的处理

**代码位置**：`tests/test_strategy.py` 第356-395行

```python
if result.success:
    # 放置成功，添加到 active_workflows
    env.active_workflows.append({...})
    record["r_t"] = float(env._compute_rt())
else:
    # 放置失败，不添加到 active_workflows
    record["success"] = False
    record["r_t"] = None  # 失败任务的r_t为None
```

**结论**：
- 失败的任务**不会**被添加到 `active_workflows`
- 失败任务的 `record["r_t"]` 为 `None`
- 失败任务**不会影响**后续时间步的r_t计算

### 2.4 训练时的penalty机制

**代码位置**：`env.py` 第549-576行

```python
def place_task(self, vn: Data, mapping: Dict[int, int], lifetime: float) -> Tuple[bool, float]:
    # 如果放置失败，返回 penalty
    if not self._check_node_feasible(vn, mapping):
        return False, self.penalty  # ← 训练时使用
    # ...
    r_t = self._compute_rt()
    return True, r_t
```

**注意**：
- `penalty` 只在训练时使用（`env.place_task()` 方法）
- 测试时使用的是 `strategy.place()` 方法，不直接调用 `env.place_task()`
- 测试时失败的任务不会影响r_t计算

---

## 三、结论

### 3.1 测试时的行为

**放置失败时**：
- ✅ 任务**不会**被添加到 `active_workflows`
- ✅ 失败任务的 `record["r_t"]` 为 `None`
- ✅ 失败任务**不会影响**后续时间步的r_t计算
- ✅ **不会对r_t进行惩罚**

### 3.2 原因

1. **r_t的定义**：r_t是基于当前存活的workflow的平均跳数计算的
2. **失败任务的处理**：失败的任务不会被添加到 `active_workflows`，所以不会影响r_t
3. **设计理念**：r_t反映的是当前系统状态（存活的workflow），而不是历史失败记录

### 3.3 与训练时的区别

**训练时**：
- 使用 `env.place_task()` 方法
- 失败时返回 `penalty`（负值，如-10.0）
- 这个penalty用于训练时的奖励信号

**测试时**：
- 使用 `strategy.place()` 方法
- 失败时只记录 `success=False`，不影响r_t
- r_t只反映当前存活的workflow的状态

---

## 四、建议

### 4.1 如果需要惩罚机制

如果希望在测试时也对失败进行惩罚，可以考虑：

1. **修改r_t计算**：在 `_compute_rt()` 中考虑失败任务数量
2. **添加失败惩罚**：在记录r_t时，如果有失败任务，减去惩罚值
3. **单独记录失败率**：通过 `acceptance_rate` 等指标来反映失败情况

### 4.2 当前设计的合理性

**当前设计是合理的**：
- r_t反映的是当前系统状态（存活的workflow）
- 失败率通过 `acceptance_rate` 单独统计
- 避免了失败任务对r_t计算的干扰

---

**文档版本**：v1.0  
**创建时间**：2025年1月

