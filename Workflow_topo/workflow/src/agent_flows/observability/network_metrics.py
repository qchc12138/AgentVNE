"""网络通信指标监控类 - 跟踪进程级别的网络IO操作"""

import os
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class NetworkMetrics:
    """网络通信指标数据类"""
    bytes_sent: int = 0           # 发送字节数
    bytes_recv: int = 0           # 接收字节数
    packets_sent: int = 0         # 发送数据包数
    packets_recv: int = 0         # 接收数据包数
    connections_count: int = 0    # 网络连接数
    sent_mb: float = 0.0          # 发送大小（MB）
    recv_mb: float = 0.0          # 接收大小（MB）
    supported: bool = False       # 是否支持进程级网络监控


class NetworkMonitor:
    """网络监控器 - 跟踪特定进程的网络通信"""

    def __init__(self, pid: Optional[int] = None):
        """
        初始化网络监控器

        Args:
            pid: 要监控的进程ID，默认为当前进程
        """
        self.pid = pid or os.getpid()
        try:
            self.process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            raise ValueError(f"进程 {self.pid} 不存在")

        # 注意：psutil在某些平台上不支持per-process的网络IO统计
        # 我们将使用系统级网络统计作为参考
        self.start_net_io = None
        self.supported = False

    def start(self):
        """开始监控"""
        try:
            # 获取系统级网络IO（不是进程级，因为大多数系统不支持）
            net_io = psutil.net_io_counters()
            self.start_net_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            self.supported = True
        except Exception:
            self.start_net_io = None
            self.supported = False

    def stop(self) -> NetworkMetrics:
        """
        停止监控并返回指标

        Returns:
            NetworkMetrics对象，包含所有网络通信相关指标
        """
        metrics = NetworkMetrics()
        metrics.supported = self.supported

        # 获取进程的网络连接数
        try:
            connections = self.process.connections(kind='inet')
            metrics.connections_count = len(connections)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            metrics.connections_count = 0

        # 由于大多数系统不支持per-process网络IO，我们只能提供系统级的增量
        # 这个值包含了所有进程的网络活动，但仍然有参考价值
        if self.supported and self.start_net_io:
            try:
                net_io = psutil.net_io_counters()
                metrics.bytes_sent = net_io.bytes_sent - self.start_net_io['bytes_sent']
                metrics.bytes_recv = net_io.bytes_recv - self.start_net_io['bytes_recv']
                metrics.packets_sent = net_io.packets_sent - self.start_net_io['packets_sent']
                metrics.packets_recv = net_io.packets_recv - self.start_net_io['packets_recv']

                metrics.sent_mb = metrics.bytes_sent / (1024 * 1024)
                metrics.recv_mb = metrics.bytes_recv / (1024 * 1024)
            except Exception:
                pass

        return metrics

    def get_dict(self) -> dict:
        """返回字典格式的指标（用于序列化）"""
        metrics = self.stop()
        return {
            'connections_count': metrics.connections_count,
            'bytes_sent': metrics.bytes_sent,
            'bytes_recv': metrics.bytes_recv,
            'sent_mb': round(metrics.sent_mb, 2),
            'recv_mb': round(metrics.recv_mb, 2),
            'total_traffic_mb': round(metrics.sent_mb + metrics.recv_mb, 2),
            'packets_sent': metrics.packets_sent,
            'packets_recv': metrics.packets_recv,
            'note': 'Network IO metrics are system-wide during workflow execution, not process-specific'
        }
