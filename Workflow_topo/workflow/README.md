# LangGraph 工作流与资源监控

基于 LangGraph 的多节点工作流，附带 CPU/内存/存储/GPU/网络的监控与结果归档。

## 目录速览

```
workflow1/
├── src/agent_flows/
│   ├── workflows/
│   │   ├── workflow1.py              # 酒店预订工作流（7 节点）
│   │   └── workflow2.py              # 城市路口交通监控工作流（6 节点）
│   ├── common/
│   │   ├── workflow1_tools.py        # 预订相关工具
│   │   ├── workflow2_tools.py        # 摄像/图像处理工具，可选 GPU 平滑
│   │   └── utils.py                  # LLM 配置
│   ├── observability/                # 监控模块（采样/节点包装）
│   ├── workflow1_runner.py           # workflow1 基础运行入口
│   ├── workflow1_runner_monitor.py   # workflow1 带监控入口
│   ├── workflow2_runner.py           # workflow2 基础运行入口
│   └── workflow2_runner_monitor.py   # workflow2 带监控入口
├── results/                          # 运行与监控输出
└── pyproject.toml
```

## 两个工作流

- workflow1：酒店预订（意图→参数→规划→搜索→筛选→支付→总结）。
- workflow2：路口交通监控（意图→采集→预处理→特征→分析→总结），支持可选 GPU 平滑（torch+CUDA 可用时自动尝试）。

## 监控框架

- 入口封装：TracedWorkflowRunner 在各 runner_monitor 中调用，自动对节点做 monkey patch，采集 CPU/内存/存储/GPU/网络。
- 采样：默认 0.5s 间隔，结果写入 results/workflowX_时间戳.json。
- GPU：通过 nvidia-smi 拉取进程级显存；若 torch+CUDA 可用，预处理阶段尝试 GPU 平滑，否则标记 gpu_skip_*。

## 快速运行

```bash
cd workflow1
uv run src/agent_flows/workflow1_runner.py          # 运行 workflow1（仅结果）
uv run src/agent_flows/workflow1_runner_monitor.py  # 运行 workflow1 并监控
uv run src/agent_flows/workflow2_runner.py          # 运行 workflow2（仅结果）
uv run src/agent_flows/workflow2_runner_monitor.py  # 运行 workflow2 并监控
```

如需自定义用户输入或监控开关，直接编辑对应 runner 文件里的配置常量。

## 开发提示

- LLM 配置：在 [src/agent_flows/common/utils.py](src/agent_flows/common/utils.py) 设置你的模型/密钥。
- 新工作流接入：新增 workflowX.py 并提供 run_xxx_workflow 接口；仿照现有 runner/runner_monitor 新增入口即可，无需改监控代码。
- GPU 可见性：若 torch.cuda.is_available() 为 True，预处理步骤会记录 gpu_smooth3x3 或 gpu_skip_*，监控 JSON 中可查看显存与利用率。

## 注意

- GPU 监控依赖 NVIDIA 驱动与 nvidia-smi。
- 网络指标为系统级总览，并非进程精确统计。
- 若存储监控受限，storage 部分可能返回不支持。


