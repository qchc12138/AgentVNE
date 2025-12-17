# prob_test_2.py 工作流程文档

## 📋 脚本概述

`prob_test_2.py` 是一个概率矩阵测试脚本，用于：
- 处理多个 workflow 的放置
- 比较预训练模型和微调模型的概率矩阵输出
- 分析模型在最后一个 workflow 上的决策差异

---

## 🔄 完整工作流程

### **阶段 0: 初始化准备**

```
1. 创建输出目录
   ├─ 基础目录: prob_test_output/
   └─ 运行目录: prob_test_output/run_YYYYMMDD_HHMMSS/
   
2. 设置模型路径
   ├─ 预训练模型: pretrain_outputs/checkpoint_latest.pt
   └─ 微调模型: finetuning_output_3/policy_network_latest.pth
   
3. 设置拓扑路径
   ├─ SN拓扑: topo/SN_topology_2.json
   └─ Workflow: workflow_topo/workflow1_topo.json
```

---

### **阶段 1: 模型加载**

```
【步骤1】加载模型
├─ 加载预训练模型 (pretrain_policy)
│   └─ 从 checkpoint_latest.pt 加载权重
│
└─ 加载微调模型 (finetuning_policy)
    └─ 从 policy_network_latest.pth 加载权重

两个模型都设置为评估模式 (eval())
```

---

### **阶段 2: 环境创建**

```
【步骤2】创建环境
├─ 创建 SimuVNEEnv
│   ├─ SN拓扑: SN_topology_2.json
│   ├─ 设备: CPU
│   ├─ 惩罚值: -150.0
│   └─ 最大任务数: num_wk (默认2)
│
├─ 重置环境 (env.reset())
│   └─ 初始化SN资源状态
│
├─ 获取SN容量 (sn_capacity)
│   └─ 用于VN特征归一化
│
└─ 创建 WorkflowGenerator
    ├─ Workflow类型: workflow1
    ├─ 到达率: 0.05
    ├─ 平均生存时间: 10.0
    └─ 随机种子: 42 (固定种子，确保可复现)
```

---

### **阶段 3: Workflow 序列处理**

```
【步骤3】处理 workflow 序列 (循环 num_wk 次，默认2次)

对于每个 workflow (wk_idx = 0, 1, ..., num_wk-1):
│
├─ 3.1 加载 Workflow
│   └─ vn = wf_gen.load_workflow_graph('workflow1')
│       └─ 返回 VN 图数据 (Data对象)
│
├─ 3.2 获取当前 SN 状态
│   └─ sn_state = env.get_sn_state()
│       └─ 包含当前所有SN节点的剩余资源
│
├─ 3.3 使用微调模型进行放置
│   │
│   ├─ 判断是否为最后一个 workflow
│   │   └─ is_last_workflow = (wk_idx == num_wk - 1)
│   │
│   ├─ 调用 place_workflow()
│   │   ├─ 输入: finetuning_policy, vn, sn_state, env
│   │   ├─ 输出: (mapping, success, finetuning_probs_tensor)
│   │   │
│   │   └─ place_workflow() 内部流程:
│   │       ├─ 应用 bias 到 SN 特征
│   │       │   └─ sn_with_bias = apply_bias_to_sn_features(sn_state, env)
│   │       │
│   │       ├─ 计算概率矩阵
│   │       │   └─ probs_matrix = policy(vn, sn_with_bias)  # [N_v, N_s]
│   │       │
│   │       ├─ 约束节点优先放置
│   │       │   └─ 如果 VN 节点有约束，优先放置到指定 SN 节点
│   │       │
│   │       ├─ 生成优先级列表（按概率降序排序）
│   │       │   └─ priority_lists = generate_priority_lists(probs_matrix)
│   │       │
│   │       ├─ BFS + k-hop 搜索放置策略
│   │       │   ├─ 选择资源需求最大的非约束 VN 节点作为起始点
│   │       │   ├─ 从优先级列表中选择 SN 节点
│   │       │   ├─ 尝试同 SN 节点放置邻居
│   │       │   └─ k-hop 搜索放置其他邻居
│   │       │
│   │       └─ 返回结果
│   │           ├─ mapping: VN节点索引 -> SN节点ID
│   │           ├─ success: 是否成功放置所有节点
│   │           └─ probs_matrix: 概率矩阵张量
│   │
│   └─ 如果是最后一个 workflow，打印详细放置信息
│
├─ 3.4 【仅最后一个 workflow】计算和保存概率矩阵
│   │
│   ├─ 【重要】保存 env 的资源状态（在 place_workflow 之前）
│   │   └─ saved_sn_resources = {sn_id: {cpu_res, mem_res, disk_res}, ...}
│   │       └─ 因为 place_workflow 会修改 env 中的资源
│   │
│   ├─ 【重要】恢复 env 的资源状态（在计算概率矩阵之前）
│   │   └─ 恢复为放置前的状态，确保与 place_workflow 使用相同的状态
│   │
│   ├─ 应用 bias 到 SN 特征（与放置时使用相同的状态）
│   │   └─ sn_with_bias = apply_bias_to_sn_features(sn_state, env)
│   │       └─ 此时 env 的资源状态已恢复为放置前的状态
│   │
│   ├─ 计算预训练模型的概率矩阵
│   │   └─ pretrain_probs = compute_probability_matrix(pretrain_policy, vn, sn_with_bias)
│   │
│   ├─ 获取微调模型的概率矩阵（从 place_workflow 返回）
│   │   └─ finetuning_probs = finetuning_probs_tensor.cpu().numpy()
│   │       └─ 这是在 place_workflow 中使用放置前状态计算的
│   │
│   ├─ 打印概率矩阵（控制台输出）
│   │   ├─ print_probability_matrix(pretrain_probs, "预训练模型")
│   │   └─ print_probability_matrix(finetuning_probs, "微调模型")
│   │
│   ├─ 保存概率矩阵到文件
│   │   ├─ pretrain_prob_matrix.txt
│   │   └─ finetuning_prob_matrix.txt
│   │
│   ├─ 计算并保存差值矩阵
│   │   ├─ diff_probs = finetuning_probs - pretrain_probs
│   │   └─ prob_matrix_diff.txt
│   │
│   └─ 保存为 NumPy 格式
│       └─ prob_matrices.npz
│           ├─ pretrain: 预训练模型概率矩阵
│           ├─ finetuning: 微调模型概率矩阵
│           └─ diff: 差值矩阵
│
├─ 3.5 检查放置结果
│   │
│   ├─ 如果放置成功 (success == True 且 len(mapping) == vn.x.size(0))
│   │   │
│   │   ├─ 计算 VN 路径和带宽需求
│   │   │   └─ vn_paths = env._compute_paths_and_bw_demand(vn, mapping)
│   │   │
│   │   ├─ 如果路径存在 (vn_paths is not None)
│   │   │   ├─ 添加到活跃 workflow 列表
│   │   │   ├─ 更新环境接受计数
│   │   │   └─ accepted_count += 1
│   │   │
│   │   └─ 如果路径不存在
│   │       ├─ 回滚资源扣减
│   │       └─ rejected_count += 1
│   │
│   └─ 如果放置失败
│       └─ rejected_count += 1
│
└─ 3.6 推进时间
    └─ env.step_time(time_delta=1.0)
```

---

### **阶段 4: 结果总结**

```
【步骤4】打印总结
├─ 总 workflow 数
├─ 成功放置数
├─ 失败数
├─ 接受率
├─ 当前活跃 workflow 数
└─ 输出目录路径
```

---

## 📊 关键数据结构

### **概率矩阵 (Probability Matrix)**

```
probs_matrix: [N_v, N_s]
├─ N_v: VN 节点数量
├─ N_s: SN 节点数量
└─ probs_matrix[i][j]: VN节点i 放置到 SN节点j 的概率
```

### **映射 (Mapping)**

```
mapping: Dict[int, int]
├─ 键: VN节点索引 (0, 1, 2, ...)
└─ 值: SN节点ID (0, 1, 2, ...)
```

### **输出文件结构**

```
run_YYYYMMDD_HHMMSS/
├─ pretrain_prob_matrix.txt      # 预训练模型概率矩阵（文本格式）
├─ finetuning_prob_matrix.txt    # 微调模型概率矩阵（文本格式）
├─ prob_matrix_diff.txt          # 差值矩阵（文本格式）
└─ prob_matrices.npz             # 所有概率矩阵（NumPy格式）
```

---

## 🔍 关键函数说明

### **place_workflow()**

放置策略的核心函数，实现：
1. **Bias 应用**: 将 `bias_cpu` 添加到 SN 节点的 CPU 特征
2. **概率计算**: 使用策略网络计算 VN 节点到 SN 节点的概率分布
3. **约束处理**: 优先处理有约束的 VN 节点
4. **优先级生成**: 按概率降序生成 SN 节点优先级列表
5. **BFS + k-hop 搜索**: 
   - 选择资源需求最大的 VN 节点作为起始点
   - 使用优先级列表选择 SN 节点
   - 尝试将邻居节点放置在同一 SN 节点
   - 如果失败，在 k-hop 邻居中搜索

### **generate_priority_lists()**

生成优先级列表（测试模式）：
- **不采样**: 直接按概率降序排序
- **确定性**: 每次运行结果相同（用于测试）

---

## ⚠️ 注意事项

1. **最后一个 Workflow 的特殊处理**:
   - 只有最后一个 workflow 会计算和保存概率矩阵
   - 只有最后一个 workflow 会打印详细的放置信息

2. **资源管理**:
   - 每个 workflow 放置时会扣减 SN 资源
   - 如果放置失败，会回滚资源扣减
   - 时间推进会移除到期的 workflow，恢复资源

3. **SN 状态一致性**（重要）:
   - `place_workflow()` 会修改 env 中的资源（扣减资源）
   - 但计算概率矩阵需要使用与 `place_workflow()` 相同的 SN 状态
   - 解决方案：
     ├─ 在调用 `place_workflow()` 之前保存 env 的资源状态
     ├─ `place_workflow()` 会修改 env 的资源
     ├─ 在计算概率矩阵之前恢复 env 的资源状态
     └─ 确保预训练和微调模型使用相同的 SN 状态（放置前的状态）

4. **Bias 应用**:
   - Bias 只在策略网络前向传播时临时应用
   - 不修改原始 SN 状态对象
   - `apply_bias_to_sn_features()` 从 env 中读取当前资源状态
   - 预训练和微调模型使用相同的 SN 状态（包含 bias）

4. **随机种子**:
   - WorkflowGenerator 使用固定种子 (42)
   - 确保不同运行时的 workflow 序列相同

---

## 📈 输出示例

```
================================================================================
【预训练模型】概率矩阵 [VN节点数=3, SN节点数=10]
================================================================================
VN节点\SN节点    SN0      SN1      SN2      SN3      SN4      SN5      SN6      SN7      SN8      SN9
      VN0    0.1234   0.2345   0.3456   0.1234   0.0567   0.0234   0.0456   0.0234   0.0123   0.0157
      VN1    0.1456   0.2567   0.2345   0.1234   0.0678   0.0345   0.0567   0.0345   0.0234   0.0234
      VN2    0.1345   0.2456   0.2234   0.1345   0.0789   0.0456   0.0456   0.0456   0.0234   0.0234
================================================================================
```

---

## 🎯 使用场景

1. **模型对比分析**: 比较预训练和微调模型的决策差异
2. **概率分布可视化**: 生成热力图（需要额外脚本）
3. **放置策略验证**: 验证微调后的模型是否改善了放置决策
4. **调试和诊断**: 分析模型在特定 workflow 上的行为

---

## 🔧 参数配置

```python
# 主要参数
num_wk = 2              # 处理的 workflow 数量
device = 'cpu'          # 计算设备
k_hop = 1              # k-hop 搜索参数

# 模型路径（硬编码）
pretrain_model_path = "..."
finetuning_model_path = "..."

# 拓扑路径（相对路径）
sn_topology_path = "topo/SN_topology_2.json"
workflow_path = "workflow_topo/workflow1_topo.json"
```

---

## 📝 总结

`prob_test_2.py` 实现了一个完整的 workflow 放置测试流程，通过处理多个 workflow 并比较预训练和微调模型的概率矩阵输出，帮助分析模型的行为差异和性能改进。

