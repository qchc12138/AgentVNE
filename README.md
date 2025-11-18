# SimuVNE: 基于强化学习的虚拟网络嵌入系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SimuVNE 是一个基于图神经网络（GNN）和强化学习（PPO）的虚拟网络嵌入（Virtual Network Embedding, VNE）解决方案。该系统通过预训练和微调两阶段训练，实现了高效的虚拟网络到物理网络的映射策略。

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [文件说明](#文件说明)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [技术架构](#技术架构)
- [实验结果](#实验结果)
- [依赖环境](#依赖环境)
- [贡献指南](#贡献指南)

## 🎯 项目概述

SimuVNE 旨在解决虚拟网络嵌入问题，即在给定的物理网络（Substrate Network, SN）上高效地放置虚拟网络（Virtual Network, VN）请求。系统采用两阶段训练策略：

1. **预训练阶段**：使用监督学习，基于 NodeRank 算法生成的数据集训练模型
2. **微调阶段**：使用 PPO 强化学习算法，在动态环境中优化策略

### 主要应用场景

- 网络功能虚拟化（NFV）
- 云计算资源调度
- 边缘计算任务放置
- 数据中心网络优化

## ✨ 核心特性

- 🧠 **图神经网络架构**：使用 GCN + Self-Attention + Transformer Encoder
- 🎓 **两阶段训练**：预训练 + PPO 微调
- ⚡ **时间驱动仿真**：支持动态任务到达和生存时间
- 📊 **特征归一化**：统一的特征归一化策略，提升训练稳定性
- 🔄 **资源竞争感知**：奖励函数考虑资源竞争和带宽共享
- 📈 **对比基准**：提供 GAL 贪心算法作为性能对比基准

## 📁 项目结构

```
SimuVNE/
├── 核心模型
│   ├── model_1.py              # 重构后的模型（推荐使用）
│   └── model.py                # 原始模型
│
├── 预训练相关
│   ├── dataset_generate.py     # 预训练数据集生成
│   ├── pretrain_1.py           # 预训练脚本（使用 model_1.py）
│   └── pretrain.py             # 原始预训练脚本
│
├── 强化学习微调
│   ├── fine_tuning_1.py        # PPO微调脚本（使用 model_1.py）
│   └── fine_tuning.py          # 原始微调脚本
│
├── 环境与训练
│   ├── env.py                  # 强化学习环境定义
│   ├── trainer.py              # 训练器类
│   └── main.py                 # 主程序入口
│
├── 测试与评估
│   ├── test.py                 # 模型测试脚本
│   └── test_2.py               # 增强测试脚本
│
├── 对比算法
│   ├── GAL.py                  # 贪心分配算法（Greedy Allocation）
│   └── GAL_2.py                # GAL算法改进版本
│
├── 配置文件
│   ├── config.json             # 训练配置文件
│   └── environment.yml         # Conda环境配置
│
├── 数据目录
│   ├── topo/                   # 物理网络拓扑
│   ├── workflow_topo/          # 虚拟网络拓扑
│   ├── pretrain_data/          # 预训练数据集
│   ├── pretrain_outputs/       # 预训练输出
│   └── finetuning_putput/      # 微调输出
│
└── 文档
    ├── README.md               # 本文件
    ├── 项目完整总结.md          # 项目详细总结
    ├── GAL_README.md           # GAL算法说明
    └── 归一化修改总结.md        # 归一化实施说明
```

## 📄 文件说明

### 核心模型文件

#### `model_1.py` ⭐ **推荐使用**
重构后的 SimuVNE 模型，包含以下改进：
- **ColumnWiseTensorNetwork**：逐列神经张量网络，实现 `hj * Wj * Hi^T`
- **Transformer Encoder**：使用 PyTorch 官方的 `nn.TransformerEncoderLayer` 替代自定义 self-attention
- **SelfAttention**：自注意力机制层，用于图节点特征编码
- **SimuVNE**：主模型类，包含 GCN、Self-Attention、NTN 和 Encoder

**关键特性**：
- 支持固定 N2（SN 节点数）的配置
- 使用 Transformer Encoder 进行矩阵编码
- 保留 hist_S 特征（虽然当前未使用）

#### `model.py`
原始 SimuVNE 模型实现，包含：
- **SelfAttention**：自注意力机制层
- **ModifiedNeuralTensorNetwork**：修改的神经张量网络
- **SimuVNE**：主模型类

**注意**：建议使用 `model_1.py`，它包含最新的架构改进。

---

### 预训练相关文件

#### `dataset_generate.py`
预训练数据集生成脚本，功能包括：
- 加载 SN 和 VN 拓扑文件
- 计算 SN 的 NodeRank 值
- 使用贪心策略放置多个 workflow
- 生成 `<VN图, SN图, 标签矩阵>` 三元组
- 特征归一化处理

**主要函数**：
- `generate_pretrain_dataset()`: 生成预训练数据集
- `_greedy_place_workflow()`: 贪心放置算法
- `_nodes_to_features()`: 节点特征提取和归一化
- `_compute_sn_noderank()`: 计算 SN NodeRank

**使用方法**：
```bash
python dataset_generate.py --num_episodes 50 --workflows_per_episode 10
```

#### `pretrain_1.py` ⭐ **推荐使用**
使用 `model_1.py` 的预训练脚本，功能包括：
- 加载预训练数据集
- 使用 MSE 损失函数进行监督学习
- 支持自动检测 SN 节点数（`num_nodes_j`）
- 保存训练检查点和训练历史
- TensorBoard 日志记录

**主要类/函数**：
- `PretrainTrainer`: 预训练器类，管理训练过程
- `load_pretrain_dataset()`: 加载预训练数据集
- `create_pretrain_dataloader()`: 创建数据加载器

**使用方法**：
```bash
python pretrain_1.py --dataset pretrain_data/pretrain_dataset.pt --num_epochs 80
```

#### `pretrain.py`
原始预训练脚本，使用 `model.py`，功能与 `pretrain_1.py` 类似。

---

### 强化学习微调文件

#### `fine_tuning_1.py` ⭐ **推荐使用**
使用 `model_1.py` 的 PPO 微调脚本，功能包括：
- **ValueNet**：价值网络，估计状态价值
- **PPOAgent**：PPO 智能体，包含策略网络和价值网络
- **BFS 放置策略**：基于优先级列表和广度优先搜索的节点放置
- **批量训练**：支持收集多个 episode 后批量更新
- **训练结果保存**：自动保存模型、训练曲线和统计信息

**主要类/函数**：
- `ValueNet`: 价值网络，使用 GCN 编码 VN 和 SN
- `PPOAgent`: PPO 智能体，包含策略更新和价值更新
- `run_ppo_episode()`: 运行单个 PPO episode
- `run_ppo_batch_training()`: 批量 PPO 训练
- `save_training_results()`: 保存训练结果

**关键特性**：
- 时间驱动的任务到达（泊松分布）
- 指数分布的生存时间
- 资源实时扣减和回滚机制
- k-hop 邻居搜索策略

**使用方法**：
```bash
python fine_tuning_1.py
```

#### `fine_tuning.py`
原始微调脚本，使用 `model.py`，功能与 `fine_tuning_1.py` 类似。

---

### 环境与训练文件

#### `env.py`
强化学习环境定义，包含：

**WorkflowGenerator**：
- 泊松到达过程生成
- 指数生存时间采样
- VN 特征归一化

**SimuVNEEnv**：
- 时间驱动仿真环境
- 资源管理（CPU、内存、磁盘、带宽）
- 路径计算和带宽需求检查
- 奖励计算（考虑资源竞争）
- 任务到达和到期处理

**主要方法**：
- `get_sn_state()`: 获取 SN 当前状态（归一化）
- `try_place_task()`: 尝试放置任务
- `_compute_rt()`: 计算即时奖励
- `compute_final_return()`: 计算最终回报
- `step_time()`: 推进时间，移除到期任务

#### `trainer.py`
通用训练器类，用于标准监督学习训练：
- 模型训练和验证
- 学习率调度
- TensorBoard 日志记录
- 检查点保存和加载

**主要类**：
- `Trainer`: 训练器类，管理训练循环

#### `main.py`
主程序入口，提供命令行接口：
- 配置加载
- 训练/测试模式切换
- 随机种子设置
- 模型创建和训练启动

---

### 测试与评估文件

#### `test.py`
模型测试脚本，功能包括：
- 加载训练好的模型
- 测试节点放置策略
- BFS 扩展放置算法
- 资源检查和回滚
- 打印详细的放置过程

**主要函数**：
- `place_with_bfs_strategy()`: BFS 放置策略实现
- `test_placement_with_model()`: 测试模型放置性能
- `find_latest_finetuning_model()`: 查找最新的微调模型

#### `test_2.py`
增强测试脚本，继承 `PPOAgent` 类：
- 更详细的测试输出
- 模型性能评估
- 与原始 PPOAgent 的兼容性测试

---

### 对比算法文件

#### `GAL.py`
贪心分配算法（Greedy Allocation），作为性能对比基准：
- 按 VN 需求强度排序
- 按 SN 剩余资源排序
- 贪心匹配策略
- 使用相同的环境接口

**主要类**：
- `GreedyAllocator`: 贪心分配器类

**主要函数**：
- `run_gal_episode()`: 运行单个 GAL episode
- `run_gal_benchmark()`: 运行 GAL 基准测试

#### `GAL_2.py`
GAL 算法的改进版本，可能包含额外的优化策略。

---

### 配置文件

#### `config.json`
训练配置文件，包含：
- 模型参数（input_dim, hidden_dim, hist_dim）
- 训练参数（learning_rate, batch_size, num_epochs）
- 数据路径配置
- 日志配置

#### `environment.yml`
Conda 环境配置文件，包含所有依赖包和版本信息。

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 使用 Conda 创建环境
conda env create -f environment.yml
conda activate AgentVNE

# 或手动安装依赖
pip install torch torch-geometric numpy networkx matplotlib tqdm
```

### 2. 数据准备

确保以下目录存在：
- `topo/SN_topology.json`: 物理网络拓扑
- `workflow_topo/workflow1_topo.json`: 虚拟网络拓扑
- `workflow_topo/workflow1_noderank.json`: VN NodeRank 值

### 3. 生成预训练数据集

```bash
python dataset_generate.py \
    --num_episodes 50 \
    --workflows_per_episode 10 \
    --output_dir pretrain_data
```

### 4. 预训练模型

```bash
python pretrain_1.py \
    --dataset pretrain_data/pretrain_dataset.pt \
    --num_epochs 80 \
    --batch_size 10 \
    --learning_rate 0.0005
```

### 5. PPO 微调

```bash
python fine_tuning_1.py \
    --policy_ckpt pretrain_outputs/checkpoint_best.pt \
    --num_updates 10 \
    --num_episodes_per_update 4
```

### 6. 测试模型

```bash
python test.py --model_path finetuning_putput/run_*/policy_network.pth
```

## 📖 使用指南

### 预训练流程

1. **生成数据集**：运行 `dataset_generate.py` 生成预训练数据
2. **训练模型**：运行 `pretrain_1.py` 进行监督学习预训练
3. **检查结果**：查看 `pretrain_outputs/` 目录中的模型和训练历史

### 微调流程

1. **加载预训练模型**：在 `fine_tuning_1.py` 中指定预训练模型路径
2. **运行微调**：执行 PPO 训练，收集多个 episode 的数据
3. **查看结果**：检查 `finetuning_putput/` 目录中的训练曲线和模型

### 对比实验

运行 GAL 算法进行性能对比：
```bash
python GAL.py
```

## 🏗️ 技术架构

### 模型架构

```
输入: VN图 [N1, 6] + SN图 [N2, 6]
  ↓
GCN编码（双GCN）
  ├─ VN: GCN → Self-Attention → Hi [N1, dim]
  └─ SN: GCN → Self-Attention → Hj [N2, dim]
  ↓
ColumnWiseTensorNetwork
  hj * Wj * Hi^T → Z [N2, N1]
  ↓
转置: Z [N1, N2]
  ↓
Transformer Encoder
  Z → Z' [N1, N2]
  ↓
Softmax (按行)
  Z' → output [N1, N2]
```

### 训练流程

1. **预训练**：监督学习，使用 NodeRank 标签
2. **微调**：PPO 强化学习，优化放置策略
3. **评估**：在测试集上评估性能

### 关键特性

- **特征归一化**：所有特征归一化到 [0, 1] 范围
- **资源管理**：实时资源扣减和回滚机制
- **奖励设计**：考虑资源竞争和带宽共享的奖励函数
- **放置策略**：BFS 扩展 + k-hop 邻居搜索

## 📊 实验结果

### 性能指标

- **接受率**：成功放置的 VN 请求比例
- **最终回报**：考虑接受率和跳数的综合指标
- **平均跳数**：VN 链路映射的平均物理跳数

### 对比结果

| 算法 | 接受率 | 最终回报 | 平均跳数 |
|------|--------|----------|----------|
| GAL | ~100% | ~0.00 | 较高 |
| PPO | >95% | >0 | 较低 |

*注：具体结果取决于任务负载和网络拓扑*

## 🔧 依赖环境

### Python 版本
- Python 3.8+

### 主要依赖
- PyTorch >= 1.10.0
- PyTorch Geometric >= 2.0.0
- NumPy >= 1.20.0
- NetworkX >= 2.6.0
- Matplotlib >= 3.3.0
- tqdm >= 4.60.0

### 可选依赖
- TensorBoard（用于可视化）

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 贡献步骤

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 版本历史

### v1.1 (当前版本)
- ✨ 重构模型架构（`model_1.py`）
- ✨ 使用 Transformer Encoder
- ✨ 改进的 NTN 实现
- 📝 完善文档和 README

### v1.0
- 🎉 初始版本发布
- ✅ 预训练和微调流程
- ✅ GAL 对比算法
- ✅ 完整的实验框架

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- PyTorch 团队提供的优秀深度学习框架
- PyTorch Geometric 提供的图神经网络工具
- 所有贡献者和使用者的反馈

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送 Pull Request

---

**项目状态**: ✅ 活跃开发中  
**最后更新**: 2025年1月  
**维护者**: SimuVNE Team

