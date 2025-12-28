# AgentVNE

基于深度强化学习和图神经网络的虚拟网络嵌入（Virtual Network Embedding）系统

## 项目简介

AgentVNE 使用两阶段训练（预训练 + PPO微调）解决虚拟网络嵌入问题，通过图神经网络（GCN）编码网络拓扑特征，学习将虚拟网络节点和链路高效映射到底层网络上。

## 系统架构

AgentVNE 采用分层架构设计，第一层为 **LLM-based Semantic Perception & Constraint Resolution（基于LLM的语义感知与约束解析）**。

### 第一层：LLM-based Semantic Perception & Constraint Resolution

第一层通过大语言模型（LLM）进行语义感知和约束解析，实现智能节点匹配与资源增强。

**核心功能：**
- 🔍 **语义感知**：分析虚拟节点(VN)的提示词，理解节点的语义需求和功能特性
- 🎯 **约束识别**：自动识别节点所需的特殊执行环境（如PCI DSS安全环境、GPU计算环境、摄像头硬件等）
- 🔗 **智能匹配**：将需要特殊环境的VN节点匹配到合适的基础网络节点(SN)
- 📊 **资源增强**：为后续的嵌入决策提供语义层面的约束和偏置信息

**实现模块：** `LLM_resource_augmentation/node_optimizer/`

该模块通过LLM分析工作流节点的提示词，判断节点是否需要特殊执行环境，并自动匹配到合适的SN节点，为第二层的嵌入决策提供语义约束。

**使用示例：**
```bash
cd LLM_resource_augmentation/node_optimizer
uv run run_optimizer.py
```

## 核心特性

- 🎯 **两阶段训练**：预训练 + PPO微调
- 🧠 **图神经网络**：GCN编码器 + 自注意力机制
- 🔄 **强化学习**：PPO算法优化策略
- 📊 **多策略支持**：贪心、遗传算法、NodeRank等基线方法

## 快速开始

### 环境配置

```bash
conda env create -f environment.yml
conda activate AgentVNE
```

### 数据准备

1. 将SN拓扑文件放在 `topo/` 目录
2. 将Workflow拓扑文件放在 `Workflow_topo/` 目录
3. 生成预训练数据集：

```bash
python dataset_generate_1.py \
    --sn_topo topo/SN_topology.json \
    --workflow_topo Workflow_topo/workflow1_topo.json \
    --workflow_noderank Workflow_topo/workflow1_noderank.json \
    --output pretrain_data/pretrain_dataset.pt \
    --workflows_per_episode 10 \
    --num_episodes 50
```

### 训练流程

**1. 预训练**
```bash
python pretrain.py \
    --data_path pretrain_data/pretrain_dataset.pt \
    --output_dir pretrain_outputs \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.001
```

**2. 微调**
```bash
python fine_tuning.py \
    --pretrain_model pretrain_outputs/checkpoint_latest.pt \
    --sn_topology topo/SN_topology.json \
    --workflow_types Workflow_topo/workflow1_topo.json \
    --output_dir finetuning_output \
    --num_episodes 1000 \
    --max_arrived_tasks 100
```

**3. 测试评估**
```bash
python tester.py \
    --sn_topology topo/SN_topology.json \
    --workflow workflow1=Workflow_topo/workflow1_topo.json \
    --strategy ga --strategy greedy --strategy pretrain --strategy finetuned \
    --parameter arrival_rate=0.25,mean_lifetime=40,max_time_steps=11000,seed=42 \
    --plot
```

## 项目结构

```
agentvne/
├── model.py                    # SimuVNE模型（策略网络）
├── env.py                      # 环境定义（SimuVNEEnv）
├── pretrain.py                 # 预训练脚本
├── fine_tuning.py              # PPO微调脚本
├── dataset_generate_1.py       # 数据集生成
├── tester.py                   # 多策略测试脚本
├── LLM_resource_augmentation/  # 第一层：LLM语义感知与约束解析
│   └── node_optimizer/         # 节点优化器（VN-SN智能匹配）
├── baselines/                  # 基线方法
├── topo/                       # SN拓扑文件
├── Workflow_topo/              # Workflow拓扑文件
├── pretrain_data/              # 预训练数据集
├── pretrain_outputs/           # 预训练模型输出
└── finetuning_output/          # 微调模型输出
```

## 模型架构

- **GCN编码器**：对VN和SN图进行特征编码
- **自注意力机制**：增强节点特征表示
- **神经张量网络**：计算VN节点到SN节点的匹配概率
- **输出层**：生成概率矩阵 [N_v, N_s]

## 支持的策略

- `ga`: 遗传算法
- `gal-vne`: 基于NodeRank的贪心算法
- `greedy`: 基于SN排序的贪心算法
- `pretrain`: 预训练模型（ft_n）
- `finetuned`: 微调模型（ft1）

## 配置说明

主要配置在 `config.json` 中，包括模型维度、训练参数等。命令行参数支持灵活配置网络拓扑、工作流类型、训练参数等。

## 许可证

MIT License

---

**注意：本项目仍在积极开发中，API可能会发生变化。**
