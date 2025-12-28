"""内存指标监控类 - 跟踪进程级别的内存使用情况"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import psutil


@dataclass
class MemoryMetrics:
    """内存指标数据类"""
    rss: int = 0              # 常驻内存大小（字节）
    vms: int = 0              # 虚拟内存大小（字节）
    percent: float = 0.0      # 进程内存占用百分比
    peak_rss: int = 0         # 峰值常驻内存（字节）
    rss_samples: List[int] = field(default_factory=list)  # RSS采样点
    system_total: int = 0     # 系统总内存（字节）
    system_available: int = 0  # 系统可用内存（字节）
    system_percent: float = 0.0  # 系统内存使用率（%）


class MemoryMonitor:
    """内存监控器 - 跟踪特定进程的内存使用情况"""

    def __init__(self, pid: Optional[int] = None):
        """
        初始化内存监控器

        Args:
            pid: 要监控的进程ID，默认为当前进程
        """
        self.pid = pid or os.getpid()
        try:
            self.process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            raise ValueError(f"进程 {self.pid} 不存在")

        self.rss_samples = []
        self.peak_rss = 0

    def start(self):
        """开始监控"""
        self.rss_samples = []
        self.peak_rss = 0
        # 初始采样
        self.sample()

    def sample(self):
        """采样内存使用量"""
        try:
            mem_info = self.process.memory_info()
            rss = mem_info.rss
            self.rss_samples.append(rss)
            if rss > self.peak_rss:
                self.peak_rss = rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def stop(self) -> MemoryMetrics:
        """
        停止监控并返回指标

        Returns:
            MemoryMetrics对象，包含所有内存相关指标
        """
        metrics = MemoryMetrics()

        # 最后一次采样
        self.sample()

        # 获取进程内存信息
        try:
            mem_info = self.process.memory_info()
            metrics.rss = mem_info.rss
            metrics.vms = mem_info.vms
            metrics.percent = self.process.memory_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            metrics.rss = 0
            metrics.vms = 0
            metrics.percent = 0.0

        # 峰值内存
        metrics.peak_rss = self.peak_rss
        metrics.rss_samples = self.rss_samples.copy()

        # 获取系统内存信息
        system_mem = psutil.virtual_memory()
        metrics.system_total = system_mem.total
        metrics.system_available = system_mem.available
        metrics.system_percent = system_mem.percent

        return metrics

    def get_dict(self) -> dict:
        """返回字典格式的指标（用于序列化）"""
        metrics = self.stop()
        return {
            'rss_mb': round(metrics.rss / (1024 * 1024), 2),
            'vms_mb': round(metrics.vms / (1024 * 1024), 2),
            'peak_rss_mb': round(metrics.peak_rss / (1024 * 1024), 2),
            'percent': round(metrics.percent, 2),
            'system_total_gb': round(metrics.system_total / (1024 ** 3), 2),
            'system_available_gb': round(metrics.system_available / (1024 ** 3), 2),
            'system_percent': round(metrics.system_percent, 2),
            'avg_rss_mb': round(sum(metrics.rss_samples) / len(metrics.rss_samples) / (1024 * 1024), 2) if metrics.rss_samples else 0.0,
            'sample_count': len(metrics.rss_samples)
        }
