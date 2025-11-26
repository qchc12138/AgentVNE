"""
遗传算法配置管理模块：用于保存和加载 GA 参数配置。

类似于 fine_tuning_1.py 的模型保存/加载机制，但 GA 保存的是超参数配置而非模型权重。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional

__all__ = [
    "GAParams",
    "save_ga_config",
    "load_ga_config",
    "find_latest_ga_config",
    "normalize_ga_config_path",
]


class GAParams:
    """
    遗传算法参数配置类。
    
    包含所有可调优的 GA 超参数。
    """

    def __init__(
        self,
        *,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 5,
        tournament_size: int = 3,
    ):
        """
        初始化 GA 参数。
        
        参数:
            population_size: 种群大小
            max_generations: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_size: 精英个体数量
            tournament_size: 锦标赛选择大小
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

    def to_dict(self) -> Dict[str, any]:
        """转换为字典格式（用于 JSON 序列化）。"""
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "tournament_size": self.tournament_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "GAParams":
        """从字典创建 GAParams 实例。"""
        return cls(
            population_size=int(data.get("population_size", 50)),
            max_generations=int(data.get("max_generations", 100)),
            crossover_rate=float(data.get("crossover_rate", 0.8)),
            mutation_rate=float(data.get("mutation_rate", 0.1)),
            elite_size=int(data.get("elite_size", 5)),
            tournament_size=int(data.get("tournament_size", 3)),
        )


DEFAULT_GA_OUTPUT_DIR = "/home/yc2/mrt/a/ga_outputs"
DEFAULT_GA_CONFIG_FILENAME = "ga_config.json"


def normalize_ga_config_path(path: str, filename: str = DEFAULT_GA_CONFIG_FILENAME) -> str:
    """
    规范化 GA 配置路径。
    
    如果 path 是目录，自动拼接 filename；如果是文件路径，直接返回。
    
    参数:
        path: 配置路径（目录或文件）
        filename: 配置文件名（当 path 是目录时使用）
    
    返回:
        规范化后的完整文件路径
    """
    if os.path.isdir(path):
        return os.path.join(path, filename)
    return path


def save_ga_config(
    params: GAParams,
    output_dir: str = DEFAULT_GA_OUTPUT_DIR,
    *,
    config_filename: str = DEFAULT_GA_CONFIG_FILENAME,
    create_timestamp_dir: bool = True,
) -> str:
    """
    保存 GA 参数配置到文件。
    
    参数:
        params: GA 参数配置对象
        output_dir: 输出目录
        config_filename: 配置文件名（默认 "ga_config.json"）
        create_timestamp_dir: 是否创建带时间戳的子目录（默认 True）
    
    返回:
        保存的配置文件完整路径
    """
    # 创建输出目录
    if create_timestamp_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        config_path = os.path.join(run_dir, config_filename)
    else:
        os.makedirs(output_dir, exist_ok=True)
        config_path = normalize_ga_config_path(output_dir, config_filename)

    # 保存配置
    config_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S") if create_timestamp_dir else None,
        "params": params.to_dict(),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"✓ GA 配置已保存: {config_path}")
    return config_path


def load_ga_config(config_path: str, *, verbose: bool = True) -> GAParams:
    """
    从文件加载 GA 参数配置。
    
    参数:
        config_path: 配置文件路径（可以是目录或文件）
        verbose: 是否打印加载信息（默认 True）
    
    返回:
        GAParams 配置对象
    
    异常:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误
    """
    # 规范化路径
    full_path = normalize_ga_config_path(config_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"GA 配置文件不存在: {full_path}")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容不同的文件格式
        if "params" in data:
            params_dict = data["params"]
        else:
            # 如果直接是参数字典
            params_dict = data

        params = GAParams.from_dict(params_dict)
        if verbose:
            print(f"✓ GA 配置已加载: {full_path}")
        return params

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"GA 配置文件格式错误: {full_path}") from e


def find_latest_ga_config(output_dir: str = DEFAULT_GA_OUTPUT_DIR) -> Optional[str]:
    """
    查找最新的 GA 配置文件。
    
    扫描 output_dir 下的所有 run_* 目录，按时间戳排序，返回最新的配置文件路径。
    
    参数:
        output_dir: 输出目录
    
    返回:
        最新配置文件的完整路径，如果不存在则返回 None
    """
    if not os.path.exists(output_dir):
        return None

    # 查找所有 run_* 目录
    run_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("run_"):
            config_path = normalize_ga_config_path(item_path)
            if os.path.exists(config_path):
                run_dirs.append((item_path, os.path.getmtime(config_path)))

    if not run_dirs:
        return None

    # 按修改时间排序，返回最新的
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    latest_dir = run_dirs[0][0]
    latest_config = normalize_ga_config_path(latest_dir)

    return latest_config if os.path.exists(latest_config) else None

