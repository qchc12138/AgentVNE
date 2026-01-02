"""工作流追踪器 - 统一管理所有监控指标，追踪工作流和节点执行"""

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .cpu_metrics import CPUMonitor
from .gpu_metrics import GPUMonitor
from .memory_metrics import MemoryMonitor
from .network_metrics import NetworkMonitor
from .storage_metrics import StorageMonitor


class WorkflowTracer:
    """
    工作流追踪器 - 泛用的资源监控和追踪系统

    支持追踪：
    - CPU使用率
    - 内存使用
    - 存储IO
    - GPU使用（如果可用）
    - 网络通信
    - 节点执行时间和资源消耗
    """

    def __init__(
        self,
        workflow_name: str = "workflow",
        enable_cpu: bool = True,
        enable_memory: bool = True,
        enable_storage: bool = True,
        enable_gpu: bool = True,
        enable_network: bool = True,
        sampling_interval: float = 0.5  # 采样间隔（秒）
    ):
        """
        初始化工作流追踪器

        Args:
            workflow_name: 工作流名称
            enable_cpu: 是否启用CPU监控
            enable_memory: 是否启用内存监控
            enable_storage: 是否启用存储IO监控
            enable_gpu: 是否启用GPU监控
            enable_network: 是否启用网络监控
            sampling_interval: 采样间隔（秒）
        """
        self.workflow_name = workflow_name
        self.enable_cpu = enable_cpu
        self.enable_memory = enable_memory
        self.enable_storage = enable_storage
        self.enable_gpu = enable_gpu
        self.enable_network = enable_network
        self.sampling_interval = sampling_interval

        self.pid = os.getpid()

        # 监控器实例
        self.cpu_monitor = None
        self.memory_monitor = None
        self.storage_monitor = None
        self.gpu_monitor = None
        self.network_monitor = None

        # 追踪数据
        self.start_time = None
        self.end_time = None
        self.node_traces = []  # 节点追踪记录
        self.current_node = None

        # 采样线程
        self.sampling_thread = None
        self.sampling_active = False

    def _init_monitors(self):
        """初始化所有监控器"""
        if self.enable_cpu:
            self.cpu_monitor = CPUMonitor(self.pid)

        if self.enable_memory:
            self.memory_monitor = MemoryMonitor(self.pid)

        if self.enable_storage:
            self.storage_monitor = StorageMonitor(self.pid)

        if self.enable_gpu:
            self.gpu_monitor = GPUMonitor(self.pid)

        if self.enable_network:
            self.network_monitor = NetworkMonitor(self.pid)

    def _start_monitors(self):
        """启动所有监控器"""
        if self.cpu_monitor:
            self.cpu_monitor.start()

        if self.memory_monitor:
            self.memory_monitor.start()

        if self.storage_monitor:
            self.storage_monitor.start()

        if self.gpu_monitor:
            self.gpu_monitor.start()

        if self.network_monitor:
            self.network_monitor.start()

    def _sampling_loop(self):
        """后台采样循环"""
        while self.sampling_active:
            if self.cpu_monitor:
                self.cpu_monitor.sample()

            if self.memory_monitor:
                self.memory_monitor.sample()

            time.sleep(self.sampling_interval)

    def start_workflow(self):
        """开始追踪工作流"""
        self.start_time = time.time()
        self._init_monitors()
        self._start_monitors()

        # 启动采样线程
        self.sampling_active = True
        self.sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.sampling_thread.start()

    def end_workflow(self) -> Dict[str, Any]:
        """
        结束追踪工作流

        Returns:
            包含所有监控数据的字典
        """
        self.end_time = time.time()

        # 停止采样线程
        self.sampling_active = False
        if self.sampling_thread:
            self.sampling_thread.join(timeout=2)

        # 收集所有指标
        workflow_metrics = {
            'workflow_name': self.workflow_name,
            'pid': self.pid,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(self.end_time).isoformat(),
            'duration_seconds': round(self.end_time - self.start_time, 4),
            'nodes': self.node_traces,
            'summary': {}
        }

        # CPU指标
        if self.cpu_monitor:
            workflow_metrics['summary']['cpu'] = self.cpu_monitor.get_dict()

        # 内存指标
        if self.memory_monitor:
            workflow_metrics['summary']['memory'] = self.memory_monitor.get_dict()

        # 存储IO指标
        if self.storage_monitor:
            workflow_metrics['summary']['storage'] = self.storage_monitor.get_dict()

        # GPU指标
        if self.gpu_monitor:
            workflow_metrics['summary']['gpu'] = self.gpu_monitor.get_dict()

        # 网络指标
        if self.network_monitor:
            workflow_metrics['summary']['network'] = self.network_monitor.get_dict()

        return workflow_metrics

    @contextmanager
    def trace_node(self, node_name: str):
        """
        上下文管理器：追踪单个节点的执行

        Args:
            node_name: 节点名称

        Usage:
            with tracer.trace_node("node1"):
                # 节点代码
                pass
        """
        # 创建节点级监控器
        node_cpu = CPUMonitor(self.pid) if self.enable_cpu else None
        node_memory = MemoryMonitor(self.pid) if self.enable_memory else None

        # 记录开始时间
        node_start = time.time()

        # 启动节点监控
        if node_cpu:
            node_cpu.start()
        if node_memory:
            node_memory.start()

        try:
            yield
        finally:
            # 记录结束时间
            node_end = time.time()

            # 收集节点指标
            node_trace = {
                'node_name': node_name,
                'start_time': datetime.fromtimestamp(node_start).isoformat(),
                'end_time': datetime.fromtimestamp(node_end).isoformat(),
                'duration_seconds': round(node_end - node_start, 4),
                'metrics': {}
            }

            if node_cpu:
                node_trace['metrics']['cpu'] = node_cpu.get_dict()
            if node_memory:
                node_trace['metrics']['memory'] = node_memory.get_dict()

            self.node_traces.append(node_trace)

    def save_results(self, output_dir: str = "results", filename: Optional[str] = None):
        """
        保存追踪结果到文件

        Args:
            output_dir: 输出目录
            filename: 文件名（可选，默认使用时间戳）
        """
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.workflow_name}_{timestamp}.json"

        # 获取完整结果
        results = self.end_workflow()

        # 保存为JSON
        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return str(output_file)


def trace_workflow(
    workflow_func: Callable,
    workflow_name: str = "workflow",
    enable_cpu: bool = True,
    enable_memory: bool = True,
    enable_storage: bool = True,
    enable_gpu: bool = True,
    enable_network: bool = True,
    save_results: bool = True,
    output_dir: str = "results"
) -> tuple[Any, Dict[str, Any]]:
    """
    装饰器函数：自动追踪工作流执行

    Args:
        workflow_func: 要追踪的工作流函数
        workflow_name: 工作流名称
        enable_*: 各项监控开关
        save_results: 是否保存结果
        output_dir: 输出目录

    Returns:
        (工作流返回值, 追踪结果字典)

    Usage:
        def my_workflow(input_data):
            return process(input_data)

        result, metrics = trace_workflow(
            lambda: my_workflow("test"),
            workflow_name="my_workflow"
        )
    """
    tracer = WorkflowTracer(
        workflow_name=workflow_name,
        enable_cpu=enable_cpu,
        enable_memory=enable_memory,
        enable_storage=enable_storage,
        enable_gpu=enable_gpu,
        enable_network=enable_network
    )

    tracer.start_workflow()

    try:
        result = workflow_func()
    finally:
        metrics = tracer.end_workflow()
        if save_results:
            tracer.save_results(output_dir=output_dir)

    return result, metrics
