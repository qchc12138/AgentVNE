"""存储IO指标监控类 - 跟踪进程级别的磁盘IO操作"""

import os
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class StorageMetrics:
    """存储IO指标数据类"""
    read_count: int = 0       # 读操作次数
    write_count: int = 0      # 写操作次数
    read_bytes: int = 0       # 读取字节数
    write_bytes: int = 0      # 写入字节数
    read_mb: float = 0.0      # 读取大小（MB）
    write_mb: float = 0.0     # 写入大小（MB）


class StorageMonitor:
    """存储IO监控器 - 跟踪特定进程的磁盘IO操作"""

    def __init__(self, pid: Optional[int] = None):
        """
        初始化存储IO监控器

        Args:
            pid: 要监控的进程ID，默认为当前进程
        """
        self.pid = pid or os.getpid()
        try:
            self.process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            raise ValueError(f"进程 {self.pid} 不存在")

        self.start_io_counters = None
        self.supported = True

    def start(self):
        """开始监控"""
        try:
            # 在Linux上可用，Windows和macOS可能不支持
            io_counters = self.process.io_counters()
            self.start_io_counters = {
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count,
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes
            }
            self.supported = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            # 某些系统不支持IO计数器
            self.start_io_counters = None
            self.supported = False

    def stop(self) -> StorageMetrics:
        """
        停止监控并返回指标

        Returns:
            StorageMetrics对象，包含所有存储IO相关指标
        """
        metrics = StorageMetrics()

        if not self.supported or self.start_io_counters is None:
            return metrics

        try:
            io_counters = self.process.io_counters()

            # 计算差值
            metrics.read_count = io_counters.read_count - self.start_io_counters['read_count']
            metrics.write_count = io_counters.write_count - self.start_io_counters['write_count']
            metrics.read_bytes = io_counters.read_bytes - self.start_io_counters['read_bytes']
            metrics.write_bytes = io_counters.write_bytes - self.start_io_counters['write_bytes']

            # 转换为MB
            metrics.read_mb = metrics.read_bytes / (1024 * 1024)
            metrics.write_mb = metrics.write_bytes / (1024 * 1024)

        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass

        return metrics

    def get_dict(self) -> dict:
        """返回字典格式的指标（用于序列化）"""
        metrics = self.stop()
        return {
            'read_count': metrics.read_count,
            'write_count': metrics.write_count,
            'read_mb': round(metrics.read_mb, 2),
            'write_mb': round(metrics.write_mb, 2),
            'total_io_mb': round(metrics.read_mb + metrics.write_mb, 2),
            'supported': self.supported
        }
