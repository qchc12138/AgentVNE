# AgentVNE 项目复现计划书

## 项目背景

复现论文 **AgentVNE: LLM-Augmented Graph Reinforcement Learning for Affinity-Aware Multi-Agent Placement in Edge Agentic AI** 中的虚拟网络嵌入（VNE）框架。

### 核心问题
在边缘计算场景下，将动态到达的工作流（Virtual Network, VN）最优地部署到物理基底网络（Substrate Network, SN）上，目标是最小化通信延迟、最大化任务接受率和资源利用率。

### 技术方案
双层架构：

| 层级 | 技术 | 作用 |
|------|------|------|
| Layer 1 | LLM 语义感知 | 分析工作流节点的功能描述，识别特殊执行环境需求（GPU、PCI DSS 安全、摄像头等），自动匹配 SN 节点 |
| Layer 2 | GCN + Transformer + NTN + PPO | 图神经网络学习 VN-SN 图相似性，强化学习优化在线放置策略 |

---

## 当前代码完整度评估

### 已完成模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 模型定义 | `model.py` | ✅ 完成 | SimuVNE 模型（GCN + Transformer + SelfAttention + NTN），输出 VN-SN 匹配概率矩阵 |
| 模型变体 | `model__sigmoid.py` | ✅ 完成 | sigmoid 版本，输出带约束的可选节点概率 |
| 数据集生成 | `dataset_generate_1.py` | ✅ 完成 | 根据拓扑和 NodeRank 标签生成预训练监督数据集 |
| 预训练 | `pretrain.py` | ✅ 完成 | PretrainTrainer，import 已修正为 `from model import SimuVNE`，含 KL+MSE loss、TensorBoard 日志 |
| RL 环境 | `env.py` | ✅ 完成 | 699 行完整实现：`WorkflowGenerator`（泊松到达+指数生存时间）+ `SimuVNEEnv`（时间驱动 RL 环境，资源管理、奖励计算、合法动作 mask） |
| PPO 微调 | `fine_tuning.py` | ✅ 完成 | `ValueNet`（双分支 GCN）、`PPOAgent`（PPO-Clip + GAE）、训练循环、可视化 |
| 评估测试 | `tester.py` | ✅ 完成 | 统一评估框架：5 种策略对比、参数扫描、生成 Figure 4/5 图表 |
| 基线策略 | `baselines/` | ✅ 完成 | 6 种策略：`GreedySN`、`GreedyNodeRank`、`GA`、`GRC`、`PretrainStrategy`、`AgentVNEStrategy` |
| SN 拓扑工具 | `topo/` | ✅ 完成 | Waxman 模型拓扑生成、NodeRank 计算、可视化和 bias 添加 |
| 工作流拓扑 | `Workflow_topo/` | ✅ 完成 | 示例工作流拓扑和 NodeRank 计算 |
| LLM 模块 | `LLM_resource_augmentation/` | ✅ 完成 | 基于 LLM 的 VN-SN 节点语义匹配系统，使用 `uv` 运行 |
| 配置文件 | `config.json` | ✅ 完成 | 模型维度和训练超参数 |
| 环境依赖 | `environment.yml` | ✅ 完成 | Conda 环境定义，含全部 Python 依赖 |

### 已完成的小规模验证

| 验证项 | 输出目录/文件 | 状态 | 说明 |
|--------|------|------|------|
| 预训练 | `pretrain_outputs_val/` | ✅ 通过 | 5 epoch 快速验证，best val loss = 0.001556，checkpoint 已保存 |
| PPO 微调 | `finetuning_output_4/` | ⚠️ 初步 | 启动了 PPO 训练流程，加载预训练 checkpoint 成功，完成少量更新 |
| 全策略评估 | `eval_output_val/` | ✅ 通过 | 6 种策略对比（greedy, noderank, ga, pretrain, agentvne, grc）全部完成，生成 Figure 4/5 图表 |

---

## 复现路线图

### Phase 1：环境搭建与数据准备 ✅ 已完成

**目标**：搭建开发环境，生成预训练数据，跑通预训练流程。

#### 子任务（均已完成验证）

1.1 **创建 Conda 环境** ✅
```bash
conda env create -f environment.yml
conda activate AgentVNE
```

1.2 **修复 import 问题** ✅
- `pretrain.py` 已修正为 `from model import SimuVNE`

1.3 **生成预训练数据集** ✅
```bash
python dataset_generate_1.py \
    --sn_topo topo/SN_topology.json \
    --workflow_topo Workflow_topo/workflow1_topo.json \
    --workflow_noderank Workflow_topo/workflow1_noderank.json \
    --output pretrain_data/pretrain_dataset.pt \
    --workflows_per_episode 10 \
    --num_episodes 50
```
- 输出：`pretrain_data/pretrain_dataset.pt`（~1.2 MB）、`pretrain_dataset_val.pt`

1.4 **运行预训练** ✅
```bash
python pretrain.py \
    --data_path pretrain_data/pretrain_dataset.pt \
    --output_dir pretrain_outputs \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.001
```
- 检查点：`pretrain_outputs_val/checkpoint_best.pt`、`checkpoint_latest.pt`
- TensorBoard 日志显示 loss 收敛

---

### Phase 2：实现 `env.py`——核心 RL 环境 ✅ 已完成

**目标**：完整实现 `SimuVNEEnv` 和 `WorkflowGenerator`。

实现位于 [env.py](/e:/E桌面/AgentVNE/env.py)（699 行）。严格按论文 Section 3（System Model）和 Algorithm 1 设计。

#### 2.1 WorkflowGenerator ✅

- **泊松到达过程**：每时间步按 `arrival_rate` 采样到达数量
- **指数分布生存时间**：新到达 VN 分配随机 lifetime
- **事件驱动队列**：维护活跃 VN 集合，管理到达/离开事件
- **多工作流类型**：`add_workflow_type(name, topo_path)` 注册多种模板
- **关键方法**：`generate_arrival_events()`, `step_time()`, `get_active_vns()`, `reset()`

#### 2.2 SimuVNEEnv ✅

- **状态空间**：SN/VN 节点特征（CPU/Memory/Disk/Bandwidth/CommBandwidth，6 维归一化）→ PyG Data 对象
- **动作空间**：策略网络输出概率矩阵 `[N_vn, N_sn]`，贪心采样选最高概率合法 SN 节点
- **奖励函数**：成功放置正奖励 + 通信延迟惩罚（最短路径长度）+ 拒绝负奖励
- **转移逻辑**：时间推进 → 释放到期 VN → 处理新到达 VN → 节点映射 → 资源分配 → 奖励计算
- **约束支持**：`legal_mask` 处理约束节点和资源不足
- **关键方法**：`reset()`, `step()`, `get_state()`, `is_valid_assignment()`, `get_legal_mask()`, `allocate_resources()`, `release_resources()`, `compute_reward()`, `_greedy_place()`

---

### Phase 3：完善 `fine_tuning.py`——PPO 微调 ✅ 已完成

**目标**：实现完整的 PPOAgent、ValueNet 和训练循环。

实现位于 [fine_tuning.py](/e:/E桌面/AgentVNE/fine_tuning.py)。

#### 3.1 ValueNet ✅
- 双分支 GCN 编码器（SN + VN）→ 全局平均池化 → MLP Head → 标量状态价值

#### 3.2 PPOAgent ✅
- 策略网络：`SimuVNE` 模型（输出 VN-SN 匹配概率矩阵）
- 价值网络：`ValueNet`
- PPO-Clip + GAE：`compute_log_prob()`, `compute_gae()`, `store_trajectory()`, `update()`
- 超参数：`clip_epsilon=0.2`, `gamma=0.99`, `gae_lambda=0.95`, `entropy_coef=0.01`, `value_loss_coef=0.5`

#### 3.3 训练循环 ✅
- `run_ppo_episode()`：单 episode 轨迹收集
- `run_ppo_batch_training()`：批量 episode 收集 + 多轮 PPO 更新
- `save_training_results()`：保存 checkpoint + 训练统计 JSON + 训练曲线图（return / acceptance / avg_rt）

---

### Phase 4：实现 `tester.py` 和基线策略 ✅ 已完成

**目标**：完整评估 AgentVNE 与基线方法的性能差异。

#### 4.1 基线策略 ✅

| 策略 | 文件 | 复杂度 | 实现 |
|------|------|--------|------|
| Greedy-SN | `baselines/greedy.py` | O(N²) | 按剩余资源排序贪心分配 |
| Greedy-NodeRank | `baselines/noderank.py` | O(N²) | 基于预计算 NodeRank 优先级 |
| GA（遗传算法） | `baselines/ga.py` | O(N³) | 种群进化搜索，精英保留+变异 |
| GRC（图卷积排序） | `baselines/grc.py` | O(N²) | 冻结预训练 GCN 模型推理 |
| Pretrain（ft_n） | `baselines/model_based.py` | O(N²) | 仅预训练 SimuVNE 推理 |
| AgentVNE（ft_1） | `baselines/model_based.py` | O(N²) | PPO 微调后 SimuVNE 推理 |

#### 4.2 评估框架 ✅ ([tester.py](/e:/E桌面/AgentVNE/tester.py))

- `run_evaluation()`：单次仿真，统计接受率/通信延迟/推理时间
- `parameter_sweep()`：跨策略+参数组合扫描
- `plot_results()`：Figure 4（接受率 vs 到达率）、Figure 5（通信延迟柱状图）

#### 4.3 已完成验证 ✅

```bash
python tester.py \
    --sn_topo topo/SN_topology.json \
    --workflow workflow1=Workflow_topo/workflow1_topo.json \
    --parameter arrival_rate=0.1,mean_lifetime=40,max_time_steps=200,seed=42 \
    --plot \
    --output-dir eval_output_val
```

- 结果：6 种策略全部通过
- 输出：`eval_output_val/evaluation_results.json`、`figure4_acceptance_rate.png`、`figure5_comm_delay.png`

---

### Phase 5：完整实验复现与分析（待执行）

**目标**：完成与论文完全对标的多组实验。

#### 5.1 预训练全量执行
当前仅完成 5 epoch 小规模验证，需执行完整 100 epoch 训练：

```bash
python pretrain.py \
    --data_path pretrain_data/pretrain_dataset.pt \
    --output_dir pretrain_outputs \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.001
```

#### 5.2 PPO 微调全量执行
当前仅完成少量更新，需执行完整微调：

```bash
python fine_tuning.py \
    --pretrain_model pretrain_outputs/checkpoint_latest.pt \
    --sn_topology topo/SN_topology.json \
    --workflow_types workflow1=Workflow_topo/workflow1_topo.json \
    --output_dir finetuning_output \
    --num_episodes_per_update 4 \
    --train_iters 5 \
    --num_updates 50 \
    --arrival_rate 0.05 \
    --mean_lifetime 10.0
```

#### 5.3 多参数评估扫描

```bash
python tester.py \
    --sn_topo topo/SN_topology.json \
    --workflow workflow1=Workflow_topo/workflow1_topo.json \
    --strategies greedy,noderank,ga,pretrain,agentvne \
    --parameter arrival_rate=0.1,0.2,0.3,0.4,0.5,mean_lifetime=20,40,60,max_time_steps=11000,seed=42 \
    --plot \
    --output-dir eval_output
```

#### 5.4 LLM 模块端到端集成
1. 运行 LLM 模块，为 workflow1 节点生成约束匹配
2. 将约束信息写入拓扑 JSON 的 `constraint_node` 字段
3. 验证 `env.py` 的 `get_legal_mask()` 正确处理约束
4. 对比有/无 LLM 约束的消融实验

#### 5.5 实验设计

| 实验 | 变量 | 预期结果 |
|------|------|----------|
| 接受率 vs 到达率 | 到达率 0.1~0.5 | AgentVNE 接受率高于基线 5%-15% |
| 通信延迟对比 | 各策略 | AgentVNE / GA 延迟显著低于 greedy/noderank |
| 推理效率对比 | 各策略 | AgentVNE 推理时间接近 greedy，远低于 GA |
| 消融实验 | 移除 LLM 层 / 移除 PPO 微调 | 验证各组件贡献 |

---

## 总时间预估（更新）

| 阶段 | 内容 | 预计时间 | 难度 | 状态 |
|------|------|----------|------|------|
| Phase 1 | 环境搭建与数据准备 | 1 天 | ⭐ | ✅ 已完成 |
| Phase 2 | 实现 RL 环境（env.py） | 5 天 | ⭐⭐⭐⭐⭐ | ✅ 已完成 |
| Phase 3 | 完善 PPO 微调 | 3 天 | ⭐⭐⭐⭐ | ✅ 已完成 |
| Phase 4 | 实现测试和基线 | 3 天 | ⭐⭐⭐ | ✅ 已完成 |
| Phase 5 | 完整实验复现与分析 | 2-3 天 | ⭐⭐ | 待执行 |

## 关键文件清单

| 文件 | 行数（约） | 作用 | 最后修改 |
|------|-----------|------|----------|
| [model.py](/e:/E桌面/AgentVNE/model.py) | ~180 | GCN+Transformer+NTN 模型 | 2026-06-13 |
| [model__sigmoid.py](/e:/E桌面/AgentVNE/model__sigmoid.py) | ~220 | 带 SelfAttention 的 sigmoid 变体 | 2026-06-13 |
| [env.py](/e:/E桌面/AgentVNE/env.py) | 699 | RL 环境 + 工作流生成器 | 2026-06-10 |
| [pretrain.py](/e:/E桌面/AgentVNE/pretrain.py) | ~700 | 预训练器（含 KL+MSE loss） | 2026-06-09 |
| [fine_tuning.py](/e:/E桌面/AgentVNE/fine_tuning.py) | ~800 | PPO 策略+价值网络+训练循环 | 2026-06-13 |
| [tester.py](/e:/E桌面/AgentVNE/tester.py) | ~450 | 多策略评估+参数扫描+绘图 | 2026-06-13 |
| [dataset_generate_1.py](/e:/E桌面/AgentVNE/dataset_generate_1.py) | ~1100 | 预训练数据集生成 | 2026-06-13 |
| [config.json](/e:/E桌面/AgentVNE/config.json) | 36 | 模型/训练超参数配置 | 2026-06-13 |
| [baselines/](/e:/E桌面/AgentVNE/baselines/) | 6 文件 | 6 种基线策略 | 2026-06-13 |
| [LLM_resource_augmentation/](/e:/E桌面/AgentVNE/LLM_resource_augmentation/) | ~10 文件 | LLM 语义匹配模块 | 2026-06-08 |

## 关键技术难点与应对

### 难点 1：NTN 模块的 SN 节点数固定约束 ✅ 已解决
- `ColumnWiseTensorNetwork` 要求固定 `num_nodes_j`。代码中通过 `config.json` 和 checkpoint 的 `model_config` 字段传递该参数
- 基线策略从 checkpoint 权重 shape 自动推断 `num_nodes_j` 实现 fallback

### 难点 2：env.py 的设计 ✅ 已解决
- 完整泊松到达 + 指数分布生命周期实现
- 资源管理含 Bandwidth 约束预留
- 通信延迟惩罚基于 NetworkX 最短路径

### 难点 3：多工作流类型处理 ✅ 已解决
- `WorkflowGenerator` 支持多种拓扑模板，按到达率随机选择
- `_greedy_place()` 处理变长 VN→SN 映射

### 难点 4：LLM 模块端到端集成 ⚠️ 部分完成
- Layer 1 LLM 模块已独立实现，输出 JSON 约束匹配结果
- 需将约束注入拓扑 JSON 实现端到端流程
- 消融实验（有/无 LLM 约束）待执行

---

## 先决条件检查清单

- [x] 完成论文精读，理解 Section 3（System Model）和 Section 4（Training Algorithm）
- [x] 安装 Conda 环境管理器
- [x] 有可用的 GPU（推荐 NVIDIA GPU + CUDA 12.x）— 当前验证使用 CPU
- [x] 具备 PyTorch 和 PyTorch Geometric 基础知识
- [x] 具备强化学习（PPO 算法）基础
- [x] 熟悉虚拟网络嵌入（VNE）基本概念
- [x] 通过语法检查和 import 测试
- [x] 预训练/微调/评估流程已在 CPU 上通过小规模验证
- [ ] 如需运行 LLM 模块，需配置 LLM API（建议使用 OpenAI 兼容接口）
- [ ] 完整规模训练和参数扫描待执行