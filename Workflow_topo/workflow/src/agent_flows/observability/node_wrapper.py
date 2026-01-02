"""节点追踪包装器 - 在不修改工作流代码的情况下追踪节点执行"""

import functools
import importlib
from typing import Any, Callable, Dict, List

from .workflow_tracer import WorkflowTracer


def create_traced_node(node_func: Callable, node_name: str, tracer: WorkflowTracer) -> Callable:
    """
    创建一个被追踪的节点函数包装器

    Args:
        node_func: 原始节点函数
        node_name: 节点名称
        tracer: 工作流追踪器

    Returns:
        包装后的节点函数
    """
    @functools.wraps(node_func)
    def traced_wrapper(*args, **kwargs):
        # 使用追踪器追踪节点执行
        with tracer.trace_node(node_name):
            return node_func(*args, **kwargs)

    return traced_wrapper


def patch_workflow_nodes(
    workflow_module_path: str,
    node_functions: Dict[str, str],
    tracer: WorkflowTracer
) -> dict:
    """
    通过 monkey patching 为工作流节点添加追踪（不修改源码）

    Args:
        workflow_module_path: 工作流模块路径（如 'src.agent_flows.workflows.workflow1'）
        node_functions: 节点名称到函数名的映射（如 {'intent': 'intent_understanding_node'}）
        tracer: 工作流追踪器

    Returns:
        原始函数的备份字典，用于恢复

    Usage:
        tracer = WorkflowTracer(workflow_name="test")

        # 定义要追踪的节点
        nodes = {
            'intent': 'intent_understanding_node',
            'extract': 'parameter_extraction_node',
            'plan': 'search_planning_node',
        }

        # 应用补丁
        backup = patch_workflow_nodes('src.agent_flows.workflows.workflow1', nodes, tracer)

        # 执行工作流
        from src.agent_flows.workflows.workflow1 import run_booking_workflow
        result = run_booking_workflow("test")

        # 恢复原始函数（可选）
        restore_workflow_nodes('src.agent_flows.workflows.workflow1', backup)
    """
    # 导入模块
    module = importlib.import_module(workflow_module_path)

    # 备份原始函数
    backup = {}

    # 为每个节点函数应用追踪包装
    for node_name, func_name in node_functions.items():
        if hasattr(module, func_name):
            original_func = getattr(module, func_name)
            backup[func_name] = original_func

            # 创建追踪包装并替换
            traced_func = create_traced_node(original_func, node_name, tracer)
            setattr(module, func_name, traced_func)

    return backup


def restore_workflow_nodes(workflow_module_path: str, backup: dict):
    """
    恢复被修改的工作流节点函数

    Args:
        workflow_module_path: 工作流模块路径
        backup: 原始函数的备份字典
    """
    module = importlib.import_module(workflow_module_path)

    for func_name, original_func in backup.items():
        setattr(module, func_name, original_func)


class TracedWorkflowRunner:
    """
    被追踪的工作流运行器 - 提供完整的节点级追踪功能

    这个类提供了一个简洁的接口来执行带追踪的工作流
    """

    def __init__(
        self,
        workflow_name: str,
        workflow_module_path: str,
        node_functions: Dict[str, str],
        enable_cpu: bool = True,
        enable_memory: bool = True,
        enable_storage: bool = True,
        enable_gpu: bool = True,
        enable_network: bool = True
    ):
        """
        初始化追踪运行器

        Args:
            workflow_name: 工作流名称
            workflow_module_path: 工作流模块路径
            node_functions: 节点名称到函数名的映射
            enable_*: 各项监控开关
        """
        self.workflow_name = workflow_name
        self.workflow_module_path = workflow_module_path
        self.node_functions = node_functions

        self.tracer = WorkflowTracer(
            workflow_name=workflow_name,
            enable_cpu=enable_cpu,
            enable_memory=enable_memory,
            enable_storage=enable_storage,
            enable_gpu=enable_gpu,
            enable_network=enable_network
        )

        self.backup = None

    def __enter__(self):
        """进入上下文，应用追踪补丁"""
        self.tracer.start_workflow()
        self.backup = patch_workflow_nodes(
            self.workflow_module_path,
            self.node_functions,
            self.tracer
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原始函数"""
        if self.backup:
            restore_workflow_nodes(self.workflow_module_path, self.backup)

    def get_metrics(self) -> dict:
        """获取追踪指标"""
        return self.tracer.end_workflow()

    def save_results(self, output_dir: str = "results", filename: str = None) -> str:
        """保存追踪结果"""
        return self.tracer.save_results(output_dir, filename)

