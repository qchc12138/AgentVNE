# AgentVNE

Layer2——基于强化学习和图神经网络的虚拟网络嵌入（Virtual Network Embedding）系统

## 📋 项目简介

AgentVNE 是一个使用深度强化学习解决虚拟网络嵌入问题的系统。该系统通过预训练和微调两阶段训练，学习如何将虚拟网络（VN）节点和链路高效地映射到底层网络（SN）上，以优化资源利用率和接受率。

### 主要特性

- 🎯 **两阶段训练**：预训练 + PPO微调
- 🧠 **图神经网络**：使用GCN编码网络拓扑特征
- 🔄 **强化学习**：PPO算法进行策略优化
- 📊 **多策略支持**：包含贪心、遗传算法等基线方法
- 🔧 **灵活配置**：支持多种网络拓扑和工作流类型

## 🏗️ 项目结构

```
agentvne/
├── model.py                 # SimuVNE模型定义（策略网络）
├── model__sigmoid.py        # 带Sigmoid激活的模型变体
├── env.py                   # 环境定义（SimuVNEEnv, WorkflowGenerator）
├── pretrain.py              # 预训练脚本
├── fine_tuning.py           # PPO微调脚本
├── dataset_generate_1.py    # 数据集生成脚本
├── tester.py                # 测试脚本（支持多种策略）
├── config.json              # 配置文件
├── environment.yml          # Conda环境配置
├── baselines/               # 基线方法
│   ├── greedy.py           # 贪心算法
│   └── genetic_algorithm/  # 遗传算法
├── topo/                    # 网络拓扑文件
├── workflow_topo/           # 工作流拓扑文件
├── pretrain_data/           # 预训练数据集
├── pretrain_outputs/        # 预训练模型输出
└── finetuning_output/       # 微调模型输出
```

## 🚀 快速开始

### 环境配置

1. **使用Conda创建环境**：
```bash
conda env create -f environment.yml
conda activate AgentVNE
```

2. **手动安装依赖**（如果Conda环境创建失败）：
```bash
pip install torch torch-geometric networkx numpy matplotlib tqdm tensorboard
```

### 数据准备

1. **准备网络拓扑文件**：
   - 将SN拓扑文件放在 `topo/` 目录下
   - 将Workflow拓扑文件放在 `workflow_topo/` 目录下

2. **生成预训练数据集**：
```bash
python dataset_generate_1.py \
    --sn_topo topo/SN_topology.json \
    --workflow_topo workflow_topo/workflow1_topo.json \
    --workflow_noderank workflow_topo/workflow1_noderank.json \
    --output pretrain_data/pretrain_dataset.pt \
    --workflows_per_episode 10 \
    --num_episodes 50
```

### 训练流程

#### 1. 预训练阶段

使用生成的数据集对模型进行预训练：

```bash
python pretrain.py \
    --data_path pretrain_data/pretrain_dataset.pt \
    --output_dir pretrain_outputs \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.001
```

#### 2. 微调阶段

使用PPO算法在真实环境中进行强化学习微调：

```bash
python fine_tuning.py \
    --pretrain_model pretrain_outputs/checkpoint_latest.pt \
    --sn_topology topo/SN_topology.json \
    --workflow_types workflow_topo/workflow1_topo.json \
    --output_dir finetuning_output \
    --num_episodes 1000 \
    --max_arrived_tasks 100
```

### 测试评估

运行测试脚本评估不同策略的性能：

```bash
python tester.py \
    --sn_topology topo/SN_topology.json \
    --workflow_types workflow_topo/workflow1_topo.json \
    --strategies greedy,ga,ft1 \
    --num_tests 10
```

## 📖 详细说明

### 模型架构

**SimuVNE模型**包含以下组件：
- **GCN编码器**：对VN和SN图进行特征编码
- **自注意力机制**：增强节点特征表示
- **神经张量网络**：计算VN节点到SN节点的匹配概率
- **输出层**：生成概率矩阵 [N_v, N_s]

### 环境说明

**SimuVNEEnv** 环境特点：
- 支持多种Workflow类型
- 泊松到达过程
- 指数生存时间分布
- 资源约束检查（CPU、内存、磁盘、带宽）
- 路径计算和带宽分配

### 训练策略

1. **预训练**：
   - 使用KL散度 + MSE损失
   - 学习NodeRank标签分布
   - 批量训练，支持验证集

2. **微调（PPO）**：
   - 策略网络：SimuVNE模型
   - 价值网络：独立的GCN编码器
   - 奖励函数：基于接受率和资源利用率
   - 支持经验回放和批量更新

### 基线方法

- **Greedy**：贪心算法，优先选择资源充足的节点
- **Genetic Algorithm (GA)**：遗传算法优化
- **NodeRank-based**：基于节点重要性的启发式方法

## ⚙️ 配置说明

### config.json

```json
{
  "model": {
    "input_dim": 6,
    "hidden_dim": 64,
    "hist_dim": 32
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "batch_size": 16,
    "num_epochs": 100
  }
}
```

### 命令行参数

主要脚本支持的命令行参数：

**dataset_generate_1.py**:
- `--sn_topo`: SN拓扑文件路径
- `--workflow_topo`: Workflow拓扑文件路径
- `--workflow_noderank`: Workflow NodeRank文件路径
- `--output`: 输出数据集路径
- `--workflows_per_episode`: 每个episode的workflow数量
- `--num_episodes`: Episode数量

**pretrain.py**:
- `--data_path`: 预训练数据集路径
- `--output_dir`: 模型输出目录
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率

**fine_tuning.py**:
- `--pretrain_model`: 预训练模型路径
- `--sn_topology`: SN拓扑文件路径
- `--workflow_types`: Workflow类型字典
- `--output_dir`: 输出目录
- `--num_episodes`: 训练episode数
- `--max_arrived_tasks`: 最大到达任务数

## 📊 输出说明

### 预训练输出

- `checkpoint_latest.pt`: 最新模型检查点
- `checkpoint_best.pt`: 最佳验证性能模型
- `training_log.txt`: 训练日志
- TensorBoard日志（如果启用）

### 微调输出

- `policy_network_latest.pth`: 最新策略网络
- `value_network_latest.pth`: 最新价值网络
- `training_log.txt`: 训练日志
- `reward_history.png`: 奖励曲线图
- `acceptance_rate_history.png`: 接受率曲线图

## 🔬 实验建议

1. **数据生成**：
   - 根据实际网络规模调整 `workflows_per_episode`
   - 确保数据集足够大以覆盖各种场景

2. **预训练**：
   - 监控验证损失，避免过拟合
   - 调整学习率和批次大小

3. **微调**：
   - 根据环境动态调整PPO超参数
   - 监控接受率和奖励趋势
   - 使用不同的随机种子进行多次实验

## 🐛 常见问题

1. **CUDA内存不足**：
   - 减小批次大小
   - 使用CPU训练（`device='cpu'`）

2. **导入错误**：
   - 确保已安装所有依赖
   - 检查Python版本（推荐3.8+）

3. **路径问题**：
   - 使用绝对路径或相对于脚本目录的路径
   - 检查文件是否存在

## 📝 引用

如果使用本项目，请引用相关论文：

```bibtex
@article{agentvne,
  title={AgentVNE: Virtual Network Embedding with Deep Reinforcement Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 📄 许可证

本项目采用 MIT 许可证。

## 👥 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请通过Issue联系。

---

**注意**：本项目仍在积极开发中，API可能会发生变化。

