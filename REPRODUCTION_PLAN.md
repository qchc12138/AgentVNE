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

### 已有模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 模型定义 | `model.py` / `model__sigmoid.py` | ✅ 完成 | SimuVNE 模型（GCN+Transformer+NTN），sigmoid 版本输出带约束的可选节点概率 |
| 数据集生成 | `dataset_generate_1.py` | ✅ 完成 | 根据拓扑和 NodeRank 标签生成预训练监督数据集 |
| 预训练 | `pretrain.py` | ⚠️ 基本完成 | 有完整的 PretrainTrainer，但 import `model_1` 应改为 `model` |
| SN 拓扑工具 | `topo/` | ✅ 完成 | 拓扑生成（Waxman 模型）、NodeRank 计算、可视化和 bias 添加 |
| 工作流拓扑 | `Workflow_topo/` | ✅ 完成 | 示例工作流拓扑和 NodeRank 计算 |
| LLM 模块 | `LLM_resource_augmentation/` | ✅ 完成 | 基于 LLM 的 VN-SN 节点语义匹配系统，用 `uv` 运行 |
| 配置文件 | `config.json` | ✅ 完成 | 模型维度和训练超参数 |
| 环境依赖 | `environment.yml` | ✅ 完成 | Conda 环境定义，含全部 Python 依赖 |

### 缺失 / 待完成模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| RL 环境 | `env.py` | ❌ 占位符 | `SimuVNEEnv` 和 `WorkflowGenerator` 未实现，是整个 RL 流程的基石 |
| PPO 微调 | `fine_tuning.py` | ❌ 框架空壳 | PPOAgent / ValueNet / run_ppo_episode 函数体为空，且依赖 `env.py` |
| 评估测试 | `tester.py` | ❌ 占位符 | 多策略对比评估脚本未实现 |
| 基线策略 | `baselines/` | ❌ 不存在 | greedy、GA、NodeRank 等基线方法目录不存在 |

---

## 复现路线图

### Phase 1：环境搭建与数据准备（预计 1 天）

**目标**：跑通预训练流程，生成可用的预训练数据集和模型 checkpoint。

#### 子任务

1.1 **创建 Conda 环境**
```bash
conda env create -f environment.yml
conda activate AgentVNE
```

1.2 **修复 import 问题**
- `pretrain.py` 第 15 行：`from model_1 import SimuVNE` → `from model import SimuVNE`

1.3 **生成预训练数据集**
```bash
python dataset_generate_1.py \
    --sn_topo topo/SN_topology.json \
    --workflow_topo Workflow_topo/workflow1_topo.json \
    --workflow_noderank Workflow_topo/workflow1_noderank.json \
    --output pretrain_data/pretrain_dataset.pt \
    --workflows_per_episode 10 \
    --num_episodes 50
```

1.4 **运行预训练**
```bash
python pretrain.py \
    --data_path pretrain_data/pretrain_dataset.pt \
    --output_dir pretrain_outputs \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.001
```

1.5 **验证预训练结果**
- 检查 `pretrain_outputs/checkpoint_latest.pt` 是否成功生成
- 检查 TensorBoard 日志中 loss 曲线是否收敛

---

### Phase 2：实现 `env.py`——核心 RL 环境（预计 5 天）

**目标**：完整实现 `SimuVNEEnv` 和 `WorkflowGenerator`，使其可被 `fine_tuning.py` 调用。

这是整个复现工程中最关键也最复杂的部分。环境需要严格按照论文 Section 3（System Model）和 Algorithm 1 设计和实现。

#### 2.1 WorkflowGenerator（工作流到达生成器）

根据论文定义，工作流按泊松过程到达，每个工作流有一个随机生存时间。

**输入**：
- `arrival_rate`：泊松到达率 λ
- `mean_lifetime`：平均生存时间（指数分布参数）
- `workflow_types`：可用工作流拓扑模板列表
- `max_arrived_tasks`：单轮最大到达任务数

**输出**：动态生成的工作流序列及到达/离开事件

**实现要点**：
- 每个时间步，按泊松过程采样到达数量
- 为新到达的工作流分配生存时间（指数分布）
- 生成到达事件列表，包含 VN 拓扑数据、约束信息
- 维护活跃工作流集合，管理其生命周期

#### 2.2 SimuVNEEnv（强化学习环境）

按照论文 Algorithm 1 的框架实现。环境状态应为图特征（以 PyG Data 对象形式），动作空间为节点匹配概率矩阵。

**状态空间（State）**：
- SN 节点特征：CPU 剩余量、Memory 剩余量、Disk 剩余量、带宽、通信带宽、邻接节点数等
- SN 链路特征：节点间连接关系和剩余带宽
- VN 节点特征：CPU 需求、Memory 需求、Disk 需求、带宽需求、通信带宽需求、约束节点标识符
- VN 链路特征：带宽需求、通信需求
- SN 节点使用历史（histogram）

**动作空间（Action）**：
- 策略网络输出概率矩阵 `[N_v, N_s]`，每个元素表示 VN 节点 i 映射到 SN 节点 j 的概率
- 按节点顺序贪心采样，每次选出概率最高的合法 SN 节点
- 约束节点由 Layer 1 的 LLM 模块预指定，在合法动作空间中被硬约束

**奖励函数（Reward）**：
- 成功放置任务 → 正奖励（基于资源利用率和接受率加权）
- 拒绝任务/资源不足 → 负奖励
- 通信延迟惩罚：基于 VN 链路映射后的 SN 路径长度计算

**转移逻辑**：
1. 检查当前时间步有无到期工作流，释放资源
2. 检查当前时间步有无新到达工作流
3. 对新到达的工作流，使用策略网络依次为每个 VN 节点选择 SN 节点
4. 检查资源约束，确认合法后分配资源
5. 计算即时奖励，更新 SN 状态

**代码结构（建议）**：
```python
class WorkflowGenerator:
    def __init__(self, arrival_rate, mean_lifetime, workflow_types, ...)
    def generate_arrival_events(self, time_step)
    def generate_lifetime(self)
    def create_vn_data(self, workflow_type)

class SimuVNEEnv:
    def __init__(self, sn_topology, workflow_generator, ...)
    def reset(self)                    # 重置环境，返回初始 SN 状态
    def get_state(self)               # 返回当前 SN 状态 + 待处理 VN
    def step(self, action)            # 执行动作，返回 (next_state, reward, done, info)
    def is_valid_assignment(self, vn_node_idx, sn_node_idx)  # 验证放置合法性
    def allocate_resources(self, vn, mapping)    # 分配 SN 资源
    def release_resources(self, vn_id)           # 释放到期工作流资源
    def compute_reward(self, ...)                # 计算奖励
    def get_legal_mask(self, vn_node_idx)        # 为每个 VN 节点生成合法 SN 节点 mask
```

**参考依据**：
- 论文 Section 3.2（System Model）定义状态、动作、奖励
- 论文 Algorithm 1（Embedding Algorithm）定义转移逻辑
- `pretrain.py` / `dataset_generate_1.py` 中已有的图特征提取方法

---

### Phase 3：完善 `fine_tuning.py`——PPO 微调（预计 3 天）

**目标**：实现完整的 PPOAgent、ValueNet 和训练循环，完成预训练模型 → 微调模型的全流程。

#### 3.1 ValueNet（价值网络）

- 输入：SN 状态图 + VN 状态图
- 架构：GCN 编码器（复用 SimuVNE 的编码部分）→ 全局平均池化 → MLP Head → 标量输出（状态价值）
- 或者：简化方案——使用注意力池化聚合全部节点特征后 MLP 输出

#### 3.2 PPOAgent

- 策略网络：`SimuVNE` 模型（输出 VN-SN 匹配概率矩阵）
- 价值网络：`ValueNet`
- 收集轨迹 → 计算 GAE 优势 → PPO-Clip loss → 多 epoch 更新
- 重要超参数：
  - `clip_epsilon = 0.2`（PPO 裁剪范围）
  - `gamma = 0.99`（折扣因子）
  - `gae_lambda = 0.95`（GAE 参数）
  - `entropy_coef = 0.01`（熵正则化系数）
  - `value_loss_coef = 0.5`（价值损失权重）

#### 3.3 PPO 训练循环

- 批量收集多个 episode 的轨迹数据
- 每个批次进行多轮 PPO 更新
- 每个更新后记录平均回报和接受率
- 定期保存模型 checkpoint 和训练曲线

#### 3.4 运行 PPO 微调

```bash
python fine_tuning.py \
    --pretrain_model pretrain_outputs/checkpoint_latest.pt \
    --sn_topology topo/SN_topology.json \
    --workflow_types Workflow_topo/workflow1_topo.json \
    --output_dir finetuning_output \
    --num_episodes 1000 \
    --max_arrived_tasks 100
```

---

### Phase 4：实现 `tester.py` 和基线策略（预计 3 天）

**目标**：完整评估 AgentVNE 与基线方法的性能差异，复现论文 Figure 4/5。

#### 4.1 实现基线策略

按论文 Section 5.1 的描述实现：

| 策略 | 算法 | 复杂度 |
|------|------|--------|
| Greedy-SN | 对 SN 节点按可用资源排序，贪心分配 | O(N²) |
| Greedy-NodeRank | 基于 NodeRank 的贪心启发式 | O(N²) |
| GA（遗传算法） | 种群进化搜索最优嵌入方案 | O(N³) |
| GRC（图卷积排序） | 基于图卷积的启发式 | O(N²) |
| Pretrain（ft_n） | 仅使用预训练模型推理 | O(N²) |
| AgentVNE（ft₁） | 预训练 + PPO 微调完整方案 | O(N²) |

#### 4.2 实现评估框架

- 统一的仿真环境接口（基于 `env.py`）
- 可对比的参数：接受率、资源利用率、通信延迟、推理时间
- 支持不同到达率（0.1~0.5）和生命周期参数扫描
- 生成论文 Figure 4（接受率随到达率变化）和 Figure 5（通信延迟对比）

#### 4.3 运行评估

```bash
python tester.py \
    --sn_topology topo/SN_topology.json \
    --workflow workflow1=Workflow_topo/workflow1_topo.json \
    --strategy ga --strategy greedy --strategy pretrain --strategy finetuned \
    --parameter arrival_rate=0.25,mean_lifetime=40,max_time_steps=11000,seed=42 \
    --plot
```

---

### Phase 5：实验复现与分析（预计 2 天）

**目标**：完成与论文完全对标的多组实验，验证各性能指标。

#### 5.1 实验设计

| 实验 | 变量 | 预期结果 |
|------|------|----------|
| 接受率 vs 到达率 | 到达率 0.1~0.5 | AgentVNE 接受率高于基线 5%-10% |
| 通信延迟对比 | 各策略 | AgentVNE 延迟 < 基线 40% |
| 消融实验 | 移除 LLM 层 / 移除 PPO 微调 | 验证各组件贡献 |

#### 5.2 论文结果对标

- Figure 4：接受率随到达率和生命周期变化的热力图 / 曲线
- Figure 5：各策略通信延迟分组柱状图
- Table 1-2：各策略在标准数据集上的定量结果

---

## 总时间预估

| 阶段 | 内容 | 预计时间 | 难度 |
|------|------|----------|------|
| Phase 1 | 环境搭建与数据准备 | 1 天 | ⭐ |
| Phase 2 | 实现 RL 环境（env.py） | 5 天 | ⭐⭐⭐⭐⭐ |
| Phase 3 | 完善 PPO 微调 | 3 天 | ⭐⭐⭐⭐ |
| Phase 4 | 实现测试和基线 | 3 天 | ⭐⭐⭐ |
| Phase 5 | 实验复现与分析 | 2 天 | ⭐⭐ |
| **总计** | | **约 14 天** | |

---

## 关键技术难点与应对

### 难点 1：env.py 的设计
- **瓶颈**：论文没有完整给出环境的状态/动作定义，需要从论文 Section 3 和 Algorithm 1 中推断
- **应对**：先实现简化版（仅考虑 CPU/Memory 约束），验证流程正确后再加入带宽和路径约束

### 难点 2：NTN 模块的 SN 节点数量固定约束
- `ColumnWiseTensorNetwork` 要求在初始化时固定 `num_nodes_j`（SN 节点数）
- **应对**：仿真时固定 SN 拓扑不变，确保训练和推理使用相同的拓扑

### 难点 3：多工作流类型同时到达的处理
- VN 节点数变化时，需要动态处理不同规模的匹配矩阵
- **应对**：用 mask 机制屏蔽不存在的节点位置，或者在批处理时 pad 到最大节点数

### 难点 4：LLM 模块集成
- Layer 1 的 LLM 模块通过离线分析生成 VN 节点的约束信息，存储在 JSON/数据集中
- **应对**：Phase 1 阶段可以先用硬编码的约束信息替代 LLM 调用，Phase 4 再集成完整的 LLM 模块

---

## 先决条件检查清单

- [ ] 完成论文精读，理解 Section 3（System Model）和 Section 4（Training Algorithm）
- [ ] 安装 Conda / Mamba 环境管理器
- [ ] 有可用的 GPU（推荐 NVIDIA GPU + CUDA 12.x）
- [ ] 具备 PyTorch 和 PyTorch Geometric 基础知识
- [ ] 具备强化学习（PPO 算法）基础
- [ ] 熟悉虚拟网络嵌入（VNE）基本概念
- [ ] 如需运行 LLM 模块，需配置 LLM API（建议使用 OpenAI 兼容接口）
