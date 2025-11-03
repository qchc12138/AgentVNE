# Fine-tuning 输出目录

本目录用于保存 PPO fine-tuning 训练的所有结果。

## 目录结构

每次训练运行会创建一个带时间戳的子目录：`run_YYYYMMDD_HHMMSS/`

```
finetuning_putput/
├── run_20240101_120000/
│   ├── training_curves.png        # 训练曲线图（4个子图）
│   ├── policy_network.pth         # 策略网络参数
│   ├── value_network.pth          # 价值网络参数
│   ├── training_stats.json        # 详细训练统计（JSON格式）
│   └── training_summary.txt       # 训练摘要（文本格式）
└── run_20240101_140000/
    └── ...
```

## 文件说明

### 1. training_curves.png
包含4个子图的训练曲线可视化：
- **左上**: 平均回报（Average Return）随更新次数的变化
- **右上**: 任务接受率（Acceptance Rate）随更新次数的变化
- **左下**: 每次更新收集的样本数量
- **右下**: 接受率和回报的综合对比（双Y轴）

### 2. policy_network.pth
保存的策略网络（SimuVNE）参数，包含：
- `model_state_dict`: 模型权重
- `model_config`: 模型配置（input_dim, hidden_dim, hist_dim）

**加载方式**:
```python
from model import SimuVNE
import torch

checkpoint = torch.load('policy_network.pth')
model = SimuVNE(
    input_dim=checkpoint['model_config']['input_dim'],
    hidden_dim=checkpoint['model_config']['hidden_dim'],
    hist_dim=checkpoint['model_config']['hist_dim']
)
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3. value_network.pth
保存的价值网络（ValueNet）参数

**加载方式**:
```python
from fine_tuning import ValueNet
import torch

checkpoint = torch.load('value_network.pth')
value_net = ValueNet()
value_net.load_state_dict(checkpoint['model_state_dict'])
```

### 4. training_stats.json
详细的训练统计数据（JSON格式），包含：
- `timestamp`: 训练时间戳
- `num_updates`: 总更新次数
- `training_stats`: 每次更新的详细统计
  - `update_idx`: 更新索引
  - `avg_return`: 平均回报
  - `avg_accepted`: 平均接受任务数
  - `avg_arrived`: 平均到达任务数
  - `total_samples`: 样本数
  - `episode_stats`: 每个episode的详细统计
- `summary`: 训练总结
  - `final_avg_return`: 最终平均回报
  - `final_acceptance_rate`: 最终接受率
  - `best_return`: 最佳回报
  - `best_acceptance_rate`: 最佳接受率
  - `total_samples`: 总样本数

### 5. training_summary.txt
人类可读的训练摘要文本，包含：
- 训练时间
- 总更新次数
- 总样本数
- 每次更新的详细进展
- 最终结果统计

## 使用示例

### 继续训练（加载已保存的模型）

```python
from fine_tuning import run_ppo_batch_training

training_stats, agent = run_ppo_batch_training(
    sn_topology_path='/path/to/SN_topology.json',
    workflow_types={'workflow1': '/path/to/workflow1_topo.json'},
    policy_ckpt='/home/zrz/SimuVNE/finetuning_putput/run_XXX/policy_network.pth',  # 加载预训练模型
    device='cpu',
    arrival_rate=0.2,
    mean_lifetime=100.0,
    max_arrived_tasks=100,
    num_updates=10
)
```

### 评估模型

```python
from model import SimuVNE
from env import SimuVNEEnv, WorkflowGenerator
import torch

# 加载模型
checkpoint = torch.load('finetuning_putput/run_XXX/policy_network.pth')
policy = SimuVNE()
policy.load_state_dict(checkpoint['model_state_dict'])
policy.eval()

# 在环境中测试
env = SimuVNEEnv(sn_topology_path='path/to/sn.json')
# ... 运行评估代码
```

## 自动保存功能

训练完成后会自动调用 `save_training_results()` 函数保存所有结果：

```python
# 在 fine_tuning.py 的 main 函数中
training_stats, agent = run_ppo_batch_training(...)

# 自动保存
save_training_results(
    training_stats=training_stats,
    policy=agent.policy,
    value_net=agent.value_net,
    output_dir='/home/zrz/SimuVNE/finetuning_putput'
)
```

## 注意事项

1. 每次训练会创建新的时间戳目录，不会覆盖历史记录
2. 图片使用 300 DPI 高分辨率保存，适合论文使用
3. JSON 文件使用 UTF-8 编码，2空格缩进
4. 模型参数只保存 state_dict，需要先创建模型实例再加载

## 训练曲线解读

- **回报变化**: 负值越小越好（因为reward公式设计）
- **接受率**: 越接近 100% 越好，80%+ 为良好
- **样本数**: 应保持稳定（= num_episodes_per_update × max_arrived_tasks）
- **综合对比**: 观察接受率提升是否伴随回报改善

