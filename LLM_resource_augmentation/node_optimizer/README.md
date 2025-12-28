# Node Optimizer - 智能节点优化器

基于 LLM 的工作流节点分析与匹配系统。通过分析虚拟节点(VN)的提示词，自动判断执行环境需求，并匹配到合适的基础网络节点(SN)。

## 项目简介

在工作流系统中，不同任务节点可能需要特定执行环境：
- 支付节点需要 PCI DSS 安全环境
- 图像采集节点需要摄像头硬件
- GPU 计算节点需要显卡服务器

本系统使用 LLM 智能分析节点提示词，自动识别环境需求并匹配到底层网络节点(SN)。

## 快速开始

### 安装

```bash
cd node_optimizer
uv sync
```

### 配置

编辑 `src/node_optimizer/common/utils.py`，配置 LLM 服务：

```python
llm = ChatOpenAI(
    model="your-model",
    api_key="your-key",
    base_url="http://your-llm-endpoint/v1"
)
```

### 运行

```bash
cd src/node_optimizer
uv run run_optimizer.py
```

## 使用示例

### 命令行
```bash
$ uv run run_optimizer.py
分析完成。工作流包含7个节点，其中1个需要特殊环境：
VN6(支付节点)已匹配到SN5(安全支付节点)。
```


## 输出结果

- **终端**：简洁的分析总结
- **JSON 文件**：完整的分析数据保存在 `results/` 目录

