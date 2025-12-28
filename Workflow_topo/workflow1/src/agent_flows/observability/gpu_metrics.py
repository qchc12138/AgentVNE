"""GPU指标监控类 - 跟踪GPU使用情况（NVIDIA GPU）"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GPUMetrics:
    """GPU指标数据类"""
    gpu_count: int = 0                          # GPU数量
    gpu_utilization: List[float] = field(default_factory=list)  # GPU利用率（%）
    gpu_memory_used: List[float] = field(default_factory=list)  # GPU显存使用（MB）
    gpu_memory_total: List[float] = field(default_factory=list)  # GPU总显存（MB）
    gpu_temperature: List[float] = field(default_factory=list)   # GPU温度（℃）
    process_gpu_memory: float = 0.0            # 当前进程GPU显存使用（MB）
    available: bool = False                     # GPU是否可用


class GPUMonitor:
    """GPU监控器 - 跟踪GPU使用情况（通过nvidia-smi）"""

    def __init__(self, pid: Optional[int] = None):
        """
        初始化GPU监控器

        Args:
            pid: 要监控的进程ID，默认为当前进程
        """
        self.pid = pid or os.getpid()
        self.available = self._check_nvidia_smi()
        self.start_metrics = None

    def _check_nvidia_smi(self) -> bool:
        """检查nvidia-smi是否可用"""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_gpu_info(self) -> dict:
        """获取GPU信息"""
        if not self.available:
            return {}

        try:
            # 查询GPU利用率、显存、温度
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True
            )

            if result.returncode != 0:
                return {}

            gpu_info = {
                'utilization': [],
                'memory_used': [],
                'memory_total': [],
                'temperature': []
            }

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 5:
                        gpu_info['utilization'].append(float(parts[1]))
                        gpu_info['memory_used'].append(float(parts[2]))
                        gpu_info['memory_total'].append(float(parts[3]))
                        gpu_info['temperature'].append(float(parts[4]))

            return gpu_info

        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return {}

    def _get_process_gpu_memory(self) -> float:
        """获取进程的GPU显存使用"""
        if not self.available:
            return 0.0

        try:
            # 查询进程的GPU显存使用
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True
            )

            if result.returncode != 0:
                return 0.0

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 2:
                        pid = int(parts[0])
                        memory = float(parts[1])
                        if pid == self.pid:
                            return memory

            return 0.0

        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return 0.0

    def start(self):
        """开始监控"""
        if self.available:
            self.start_metrics = self._get_gpu_info()

    def stop(self) -> GPUMetrics:
        """
        停止监控并返回指标

        Returns:
            GPUMetrics对象，包含所有GPU相关指标
        """
        metrics = GPUMetrics()
        metrics.available = self.available

        if not self.available:
            return metrics

        # 获取GPU信息
        gpu_info = self._get_gpu_info()
        if gpu_info:
            metrics.gpu_count = len(gpu_info.get('utilization', []))
            metrics.gpu_utilization = gpu_info.get('utilization', [])
            metrics.gpu_memory_used = gpu_info.get('memory_used', [])
            metrics.gpu_memory_total = gpu_info.get('memory_total', [])
            metrics.gpu_temperature = gpu_info.get('temperature', [])

        # 获取进程GPU显存使用
        metrics.process_gpu_memory = self._get_process_gpu_memory()

        return metrics

    def get_dict(self) -> dict:
        """返回字典格式的指标（用于序列化）"""
        metrics = self.stop()

        if not metrics.available:
            return {
                'available': False,
                'message': 'GPU not available or nvidia-smi not found'
            }

        result = {
            'available': True,
            'gpu_count': metrics.gpu_count,
            'process_gpu_memory_mb': round(metrics.process_gpu_memory, 2)
        }

        # 添加每个GPU的信息
        if metrics.gpu_count > 0:
            result['gpus'] = []
            for i in range(metrics.gpu_count):
                gpu_data = {
                    'id': i,
                    'utilization_percent': round(metrics.gpu_utilization[i], 2) if i < len(metrics.gpu_utilization) else 0,
                    'memory_used_mb': round(metrics.gpu_memory_used[i], 2) if i < len(metrics.gpu_memory_used) else 0,
                    'memory_total_mb': round(metrics.gpu_memory_total[i], 2) if i < len(metrics.gpu_memory_total) else 0,
                    'temperature_c': round(metrics.gpu_temperature[i], 2) if i < len(metrics.gpu_temperature) else 0
                }
                result['gpus'].append(gpu_data)

            # 汇总信息
            if metrics.gpu_utilization:
                result['avg_utilization_percent'] = round(sum(metrics.gpu_utilization) / len(metrics.gpu_utilization), 2)
            if metrics.gpu_memory_used and metrics.gpu_memory_total:
                total_used = sum(metrics.gpu_memory_used)
                total_memory = sum(metrics.gpu_memory_total)
                result['total_memory_used_mb'] = round(total_used, 2)
                result['total_memory_mb'] = round(total_memory, 2)

        return result
