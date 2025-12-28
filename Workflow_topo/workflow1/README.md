# LangGraph 工作流监控系统

基于 LangGraph 构建的智能工作流系统，集成完整的资源监控框架。

## 项目简介

本项目包含两部分：
1. **工作流系统**：使用 LangGraph 构建的多节点智能工作流
2. **监控框架**：实时追踪 CPU、内存、存储IO、GPU、网络等资源消耗

## 目录结构

```
workflow3/
├── src/agent_flows/
│   ├── workflows/              # 工作流定义
│   │   └── workflow1.py        # 示例工作流：酒店预订（7节点）
│   ├── observability/          # 监控模块
│   │   ├── workflow_tracer.py  # 核心追踪器
│   │   ├── node_wrapper.py     # 节点包装器
│   │   ├── cpu_metrics.py      # CPU监控
│   │   ├── memory_metrics.py   # 内存监控
│   │   ├── storage_metrics.py  # 存储IO监控
│   │   ├── gpu_metrics.py      # GPU监控
│   │   └── network_metrics.py  # 网络监控
│   ├── common/                 # 公共模块
│   │   ├── tools.py            # 工具函数（搜索、支付等）
│   │   └── utils.py            # LLM配置
│   ├── runner2.py              # 带监控的执行器
│   └── runner.py               # 基础执行器
├── results/                    # 监控结果输出目录
└── pyproject.toml              # 项目依赖
```

## 核心文件说明

### 工作流模块 (workflows/)

**workflow1.py**
- 智能管家酒店预订工作流示例
- 包含7个节点：意图理解、参数提取、搜索规划、酒店搜索、筛选选择、支付、总结
- 基于 LangGraph StateGraph 构建
- 支持条件路由和状态传递

### 监控模块 (observability/)

**workflow_tracer.py**
- 核心追踪器，统一管理所有监控指标
- 支持工作流级和节点级追踪
- 后台线程定期采样（默认0.5秒）
- 提供 `start_workflow()`, `end_workflow()`, `trace_node()` 等接口

**node_wrapper.py**
- 节点级追踪包装器
- 自动包装工作流节点函数
- 追踪每个节点的资源消耗

**cpu_metrics.py**
- 进程CPU使用率（平均、峰值）
- 系统CPU使用率
- 用户态/内核态时间

**memory_metrics.py**
- 常驻内存(RSS)、虚拟内存(VMS)
- 峰值内存、内存占用百分比

**storage_metrics.py**
- 磁盘读写次数和字节数
- 总IO统计（MB）

**gpu_metrics.py**
- GPU利用率、显存使用、温度
- 支持多GPU
- 通过 nvidia-smi 获取数据

**network_metrics.py**
- 网络连接数、收发流量
- 数据包统计
- 注：系统级统计，非进程级

### 公共模块 (common/)

**tools.py**
- 工具函数：搜索酒店、预订支付、检查可用性
- 模拟数据库和业务逻辑

**utils.py**
- LLM 配置（ChatOpenAI）
- 全局 LLM 实例

### 执行器 (runner2.py)

- 可配置的工作流执行器
- 集成监控功能
- 实时输出监控摘要
- 保存详细 JSON 报告到 results/ 目录

## 快速开始

### 安装

```bash
cd workflow3
uv sync
```

### 配置 LLM

编辑 `src/agent_flows/common/utils.py`，配置你的 LLM 地址和密钥。

### 运行

```bash
cd src/agent_flows
uv run runner2.py
```

## 使用方式

### 方式1：使用 runner2.py

修改 `runner2.py` 中的配置：

```python
MONITORING_CONFIG = {
    'enable_cpu': True,
    'enable_memory': True,
    'enable_storage': True,
    'enable_gpu': True,
    'enable_network': True,
}

WORKFLOW_NAME = "workflow1"
USER_INPUT = "你的输入"
```

### 方式2：代码集成

```python
from src.agent_flows.observability import WorkflowTracer

tracer = WorkflowTracer(workflow_name="my_workflow", enable_cpu=True, enable_memory=True)
tracer.start_workflow()

result = your_workflow_function()

metrics = tracer.end_workflow()
tracer.save_results()
```

### 方式3：节点级追踪

```python
with tracer.trace_node("节点名称"):
    # 节点代码
    pass
```

## 监控指标

- **CPU**：进程使用率、系统使用率、用户态/内核态时间
- **内存**：RSS、VMS、峰值、占用百分比
- **存储IO**：读写次数、读写字节数
- **GPU**：利用率、显存、温度（需要NVIDIA驱动）
- **网络**：连接数、收发流量（系统级）

## 输出结果

- **终端输出**：配置信息、工作流结果、资源统计摘要、节点详情
- **JSON报告**：`results/工作流名称_时间戳.json`，包含完整的监控数据

## 适配新工作流

监控系统是泛用的，适配新工作流只需修改 `runner2.py` 的导入：

```python
from your_module import your_workflow_function
result = your_workflow_function(input_data)
```

无需修改工作流代码本身。


## 注意事项

- GPU 监控需要 NVIDIA 驱动和 nvidia-smi 命令
- 存储IO 监控在某些系统可能不支持（权限限制）
- 网络监控为系统级统计，非进程精确值


