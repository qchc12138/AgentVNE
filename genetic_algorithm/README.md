# 遗传算法模块 (Genetic Algorithm for VNE)

本模块实现了用于解决 VNE（Virtual Network Embedding）问题的遗传算法。

## 模块结构

- `ga_core.py`: 遗传算法核心实现
  - `Individual`: 个体（染色体）类，表示一个 VN 到 SN 的映射方案
  - `GeneticAlgorithm`: 遗传算法主类，包含选择、交叉、变异等操作
- `ga_config.py`: 配置管理模块
  - `GAParams`: GA 参数配置类
  - `save_ga_config()`: 保存配置到文件
  - `load_ga_config()`: 从文件加载配置
  - `find_latest_ga_config()`: 查找最新配置

## 算法流程

1. **初始化种群**：随机生成多个映射方案（个体）
2. **评估适应度**：对每个个体计算适应度值
   - 考虑资源约束满足度
   - 考虑路径长度（跳数）
   - 考虑资源利用率
3. **选择**：使用锦标赛选择保留优秀个体
4. **交叉**：部分映射交换生成新个体
5. **变异**：随机改变部分映射
6. **迭代**：重复步骤 2-5 直到达到最大代数

## 适应度函数

适应度函数综合考虑以下因素：

- **资源约束满足度**：不满足约束的个体给予大惩罚（-10000.0）
- **资源利用率**：CPU、内存、磁盘的平均利用率（越高越好）
- **路径长度**：所有 VN 链路对应的 SN 路径总跳数（越少越好）

最终适应度 = 10.0 × 资源利用率 - 5.0 × 归一化路径长度

## 使用方式

### 方式 1：训练并保存配置（推荐）

类似于 `fine_tuning_1.py`，可以预先训练（调优）GA 参数并保存：

1. **训练 GA 配置**：
   ```bash
   python train_ga.py
   ```
   这会测试多组参数组合，找到最优配置并保存到 `ga_outputs/run_YYYYMMDD_HHMMSS/ga_config.json`

2. **在测试时自动加载最新配置**：
   在 `tester4.py` 中，GA 策略默认会自动加载最新训练的配置：
   ```python
   strategy_names = ["null", "random", "gal", "gal_node", "ga"]
   ```
   运行 `python tester4.py` 时，GA 会自动使用 `ga_outputs/` 下最新的配置。

### 方式 2：手动指定配置路径

如果需要使用特定的配置文件：

```python
# 在 tester4.py 的 resolve_strategies 中
"ga": lambda: ga_strategy_factory(
    config_path="/path/to/ga_outputs/run_20250101_120000/ga_config.json",
    use_latest=False,  # 不使用自动查找
)
```

### 方式 3：使用默认参数（不加载配置）

如果不想使用保存的配置，可以禁用自动加载：

```python
"ga": lambda: ga_strategy_factory(
    use_latest=False,  # 禁用自动加载
    population_size=100,
    max_generations=200,
    # ... 其他参数
)
```

## 参数说明

- **population_size** (默认 50): 种群大小，影响搜索空间和计算时间
- **max_generations** (默认 100): 最大迭代代数，影响搜索深度
- **crossover_rate** (默认 0.8): 交叉概率，控制新个体生成方式
- **mutation_rate** (默认 0.1): 变异概率，增加种群多样性
- **elite_size** (默认 5): 精英个体数量，直接保留到下一代
- **tournament_size** (默认 3): 锦标赛选择大小，影响选择压力

## 性能特点

- **优点**：
  - 全局搜索能力强，能找到较好的解
  - 适应度函数可灵活设计
  - 适合作为对比基线

- **缺点**：
  - 计算时间较长（需要多代进化）
  - 不适合实时场景
  - 参数调优需要经验

## 配置保存与加载（类似 fine_tuning）

### 训练阶段
运行 `python train_ga.py` 调优并保存最优参数配置：
- 测试多组参数组合
- 找到最优配置
- 保存到 `ga_outputs/run_YYYYMMDD_HHMMSS/ga_config.json`

### 测试阶段
在 `tester4.py` 中使用 GA 策略时，默认自动加载最新配置：
- 策略初始化时自动查找并加载最新配置（`use_latest=True`）
- 使用配置中的最优参数（种群大小、迭代次数等）
- 每个 VN 任务仍会运行 GA 进化过程（这是正常的，因为每个任务都不同）

### 重要说明
**GA 与神经网络模型的区别**：
- **神经网络模型**：预训练权重 → 测试时直接使用（无需重新训练）
- **遗传算法**：预训练超参数 → 测试时使用这些参数运行 GA（每个任务都需要实时优化）

GA 每次任务都会运行进化过程是**正常且必要的**，因为：
1. 每个 VN 任务都不同（节点数、资源需求、拓扑结构）
2. SN 资源状态实时变化
3. 需要针对当前状态找到最优映射

通过预训练找到的**最优超参数**（种群大小、迭代次数等）可以：
- 提高搜索效率
- 平衡计算时间和解的质量
- 避免每次手动调参

## 注意事项

1. GA 策略计算时间较长，建议在对比实验时适当调整 `max_generations` 和 `population_size` 以平衡性能和效果
2. 如果 VN 节点数较多或 SN 网络较大，建议增加种群大小和迭代代数
3. 适应度函数可根据实际需求调整权重
4. 配置加载是自动的，无需手动指定（除非需要特定配置）

