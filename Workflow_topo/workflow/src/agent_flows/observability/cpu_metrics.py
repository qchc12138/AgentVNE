"""CPU指标监控类 - 跟踪进程级别的CPU使用情况"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import psutil


@dataclass
class CPUMetrics:
    """CPU指标数据类"""
    process_cpu_percent: float = 0.0  # 进程CPU使用率（%）
    system_cpu_percent: float = 0.0   # 系统整体CPU使用率（%）
    cpu_time_user: float = 0.0        # 用户态CPU时间（秒）
    cpu_time_system: float = 0.0      # 内核态CPU时间（秒）
    cpu_count: int = 0                # CPU核心数
    samples: List[float] = field(default_factory=list)  # CPU使用率采样点


class CPUMonitor:
    """CPU监控器 - 跟踪特定进程的CPU使用情况"""

    def __init__(self, pid: Optional[int] = None):
        """
        初始化CPU监控器

        Args:
            pid: 要监控的进程ID，默认为当前进程
        """
        self.pid = pid or os.getpid()
        try:
            self.process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            raise ValueError(f"进程 {self.pid} 不存在")

        self.start_time = None
        self.cpu_samples = []
        self.start_cpu_times = None

    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.cpu_samples = []

        # 记录初始CPU时间
        try:
            cpu_times = self.process.cpu_times()
            self.start_cpu_times = {
                'user': cpu_times.user,
                'system': cpu_times.system
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.start_cpu_times = {'user': 0, 'system': 0}

    def sample(self):
        """采样CPU使用率"""
        try:
            # 获取进程CPU使用率（需要有一定时间间隔才准确）
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.cpu_samples.append(0.0)

    def stop(self) -> CPUMetrics:
        """
        停止监控并返回指标

        Returns:
            CPUMetrics对象，包含所有CPU相关指标
        """
        metrics = CPUMetrics()

        # 获取CPU核心数
        metrics.cpu_count = psutil.cpu_count(logical=True)

        # 获取系统整体CPU使用率
        metrics.system_cpu_percent = psutil.cpu_percent(interval=0.1)

        # 计算进程CPU使用率
        if self.cpu_samples:
            metrics.process_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples)
            metrics.samples = self.cpu_samples.copy()
        else:
            # 如果没有采样，获取一次
            try:
                metrics.process_cpu_percent = self.process.cpu_percent(interval=0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                metrics.process_cpu_percent = 0.0

        # 获取CPU时间
        try:
            cpu_times = self.process.cpu_times()
            if self.start_cpu_times:
                metrics.cpu_time_user = cpu_times.user - self.start_cpu_times['user']
                metrics.cpu_time_system = cpu_times.system - self.start_cpu_times['system']
            else:
                metrics.cpu_time_user = cpu_times.user
                metrics.cpu_time_system = cpu_times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            metrics.cpu_time_user = 0.0
            metrics.cpu_time_system = 0.0

        return metrics

    def get_dict(self) -> dict:
        """返回字典格式的指标（用于序列化）"""
        metrics = self.stop()
        return {
            'process_cpu_percent': round(metrics.process_cpu_percent, 2),
            'system_cpu_percent': round(metrics.system_cpu_percent, 2),
            'cpu_time_user': round(metrics.cpu_time_user, 4),
            'cpu_time_system': round(metrics.cpu_time_system, 4),
            'cpu_count': metrics.cpu_count,
            'avg_cpu_percent': round(sum(metrics.samples) / len(metrics.samples), 2) if metrics.samples else 0.0,
            'peak_cpu_percent': round(max(metrics.samples), 2) if metrics.samples else 0.0,
            'sample_count': len(metrics.samples)
        }
