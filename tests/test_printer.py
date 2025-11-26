from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from env import SimuVNEEnv

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore

__all__ = ["TestPrinter", "DetailedTestLogger"]

TEST_SCOPE_DIRS = {
    # 单策略小参数测试（各策略独立文件夹）
    "null_single": "null_single_outs",
    "gal_single": "gal_single_outs",
    "gal2_single": "gal2_single_outs",
    "gal3_single": "gal3_single_outs",
    "ga_single": "ga_single_outs",
    "finetuned_single": "finetuned_single_outs",
    # tester 对比实验
    "tester": "tester_outs",
    "comparison": "tester_outs",
}


#region 工具函数
def _default_output_dir() -> Path:
    """默认输出目录（tests/outs），与 rewritePlan 要求保持一致。"""

    return Path(__file__).resolve().parent / "outs"


def _resolve_scope_dir(test_scope: str) -> str:
    """
    根据测试 scope 计算子目录名称。

    已知 scope 走固定映射；未知 scope 自动追加 “_outs”，便于未来扩展。
    """

    if not test_scope:
        return TEST_SCOPE_DIRS["tester"]
    if test_scope in TEST_SCOPE_DIRS:
        return TEST_SCOPE_DIRS[test_scope]
    normalized = test_scope.strip().lower().replace(" ", "_").rstrip("/")
    return normalized if normalized.endswith("_outs") else f"{normalized}_outs"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:  # pragma: no cover - 降级到 str
        return str(value)


def _format_int(value: Any) -> str:
    try:
        return f"{int(value)}"
    except Exception:  # pragma: no cover
        return str(value)


def _denormalize_task_info(task_info: Dict[str, Any], env: SimuVNEEnv) -> Dict[str, Any]:
    """
    将 VN 节点的归一化资源需求还原为绝对值，用于详细日志输出。
    """

    capacities = env.get_sn_max_capacity()
    cpu_max = capacities.get("cpu_max", 1.0)
    mem_max = capacities.get("mem_max", 1.0)
    disk_max = capacities.get("disk_max", 1.0)

    denorm_nodes: List[Dict[str, Any]] = []
    for node in task_info.get("vn_nodes_detail", []):
        node_copy = dict(node)
        node_copy["cpu"] = float(node_copy.get("cpu", 0.0)) * cpu_max
        node_copy["mem"] = float(node_copy.get("mem", 0.0)) * mem_max
        node_copy["disk"] = float(node_copy.get("disk", 0.0)) * disk_max
        denorm_nodes.append(node_copy)

    new_task_info = dict(task_info)
    new_task_info["vn_nodes_detail"] = denorm_nodes
    return new_task_info


#endregion 工具函数


class TestPrinter:
    """
    测试输出统一入口：负责汇总每轮配置、打印结果、可选日志与绘图。
    """

    def __init__(
        self,
        *,
        enable_logging: bool = False,
        enable_plotting: bool = False,
        output_dir: Optional[str] = None,
        session_name: Optional[str] = None,
        test_scope: str = "tester",
    ) -> None:
        self._columns: Sequence[Tuple[str, str]] = (
            ("策略", "strategy"),
            ("接受率(%)", "acceptance_rate"),
            ("平均r_t", "avg_r_t"),
            ("平均跳数", "avg_hops"),
            ("最大跳数", "max_hops"),
            ("平均完成时长", "avg_completion_duration"),
            ("任务数", "tasks"),
            ("接受数", "accepted"),
            ("结束时间", "end_time"),
            ("滞后时间", "lag_time"),
        )

        self._enable_logging = enable_logging  # 是否启用日志记录
        self._enable_plotting = enable_plotting and HAS_MATPLOTLIB  # 是否启用绘图（且需要matplotlib支持）
        root = Path(output_dir) if output_dir else _default_output_dir()  # 输出目录设置
        root = _ensure_dir(root)  # 创建输出目录（如不存在）

        scope_folder = _resolve_scope_dir(test_scope)  # 根据scope解析子目录名
        self._base_dir = _ensure_dir(root / scope_folder)  # 测试主输出目录
        self._detail_root = _ensure_dir(self._base_dir / "detailed_logs")  # 详细日志目录
        self._session_dir: Optional[str] = None  # 会话目录初始化为None
        self._log_file: Optional[Any] = None  # 日志文件句柄初始化为None
        self._active_detail_logger: Optional["DetailedTestLogger"] = None  # 当前详细日志记录器

        self._table_title: Optional[str] = None  # 表格标题
        self._config_info: Dict[str, Any] = {}  # 配置参数（字典）
        self._rows: List[Dict[str, Any]] = []  # 每轮汇总的结果
        self._round_logs: List[Dict[str, Any]] = []  # 每轮详细日志

        if enable_logging or self._enable_plotting:
            session_name = session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
            session_dir = _ensure_dir(self._base_dir / session_name)
            self._session_dir = str(session_dir)
            if enable_logging:
                log_path = session_dir / "test_log.txt"
                self._log_file = open(log_path, "w", encoding="utf-8")
                self._write_log_header()

    #region 属性
    @property
    def session_dir(self) -> Optional[str]:
        return self._session_dir

    #endregion 属性

    #region 生命周期管理
    def start_round(self, *, table_title: str, config_info: Dict[str, Any]) -> None:
        """开始新一轮测试输出。"""

        self._table_title = table_title
        self._config_info = dict(config_info)
        self._rows = []

        if self._log_file:
            self._log_file.write(f"\n{'='*80}\n")
            self._log_file.write(f"轮次: {table_title}\n")
            self._log_file.write(f"开始时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            self._log_file.write("测试配置:\n")
            for key, value in self._config_info.items():
                self._log_file.write(f"  {key}: {value}\n")
            self._log_file.write(f"{'='*80}\n")
            self._log_file.flush()

    def add_row(self, row: Dict[str, Any], *, strategy_info: Optional[Dict[str, Any]] = None) -> None:
        """追加策略结果，并写入日志。"""

        self._rows.append(dict(row))
        if not self._log_file:
            return

        name = row.get("strategy", "unknown")
        self._log_file.write(f"\n策略: {name}\n")
        if strategy_info:
            self._log_file.write("策略信息:\n")
            for key, value in strategy_info.items():
                self._log_file.write(f"  {key}: {value}\n")
        self._log_file.write("结果:\n")
        for title, key in self._columns:
            self._log_file.write(f"  {title}: {row.get(key, 'N/A')}\n")
        self._log_file.flush()

    def finalize(self) -> None:
        """结束当前轮次：打印、写文件、绘图。"""

        if not self._table_title:
            return

        self._print_config()
        self._print_table()
        self._store_round()

        if self._log_file:
            self._log_file.write(f"\n结束时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            self._log_file.write(f"{'-'*80}\n")
            self._log_file.flush()

        self._table_title = None
        self._config_info = {}
        self._rows = []

    def close(self) -> None:
        """会话收尾：写 summary、绘制总览、关闭日志。"""

        if self._round_logs and self._session_dir:
            session = Path(self._session_dir)
            summary_path = session / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_dir": self._session_dir,
                        "total_rounds": len(self._round_logs),
                        "rounds": self._round_logs,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            if self._enable_plotting:
                self._plot_summary(session)

        if self._log_file:
            self._log_file.write("测试完成。\n")
            self._log_file.close()
            self._log_file = None

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    #endregion 生命周期管理

    #region 打印与持久化
    def _print_config(self) -> None:
        if not self._config_info:
            return
        print(f"\n[{self._table_title}] 配置:")
        for key, value in self._config_info.items():
            print(f"  {key}: {value}")

    def _print_table(self) -> None:
        if not self._rows:
            print("当前轮次无策略结果。")
            return

        table = [[self._format_cell(row, key) for _, key in self._columns] for row in self._rows]
        headers = [title for title, _ in self._columns]

        if tabulate:
            print(tabulate(table, headers=headers, tablefmt="grid", stralign="center"))
            return

        print("\t".join(headers))
        for line in table:
            print("\t".join(line))

    def _format_cell(self, row: Dict[str, Any], key: str) -> str:
        value = row.get(key, 0)
        if key in {"strategy"}:
            return str(value)
        if key in {"tasks", "accepted"}:
            return _format_int(value)
        if key == "acceptance_rate":
            return _format_float(value, digits=2)
        return _format_float(value)

    def _store_round(self) -> None:
        if not self._rows:
            return

        round_data = {
            "round_title": self._table_title,
            "config": self._config_info.copy(),
            "results": self._rows.copy(),
            "timestamp": datetime.now().isoformat(),
        }
        self._round_logs.append(round_data)

        if not self._session_dir:
            return

        round_idx = len(self._round_logs)
        round_path = Path(self._session_dir) / f"round_{round_idx}.json"
        with open(round_path, "w", encoding="utf-8") as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)

        if self._enable_plotting:
            self._plot_round(Path(self._session_dir), round_idx, self._rows)

    #endregion 打印与持久化

    #region 绘图
    def _plot_round(self, session_dir: Path, round_idx: int, rows: List[Dict[str, Any]]) -> None:
        if not HAS_MATPLOTLIB or not rows:
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            fig.suptitle(f"{self._table_title} - 策略对比", fontsize=14)

            strategies = [row.get("strategy", "unknown") for row in rows]

            self._bar_chart(
                axes[0, 0],
                strategies,
                [row.get("acceptance_rate", 0.0) for row in rows],
                title="接受率(%)",
            )
            self._bar_chart(
                axes[0, 1],
                strategies,
                [row.get("avg_r_t", 0.0) for row in rows],
                title="平均r_t",
                color="tab:green",
            )
            self._bar_chart(
                axes[1, 0],
                strategies,
                [row.get("avg_hops", 0.0) for row in rows],
                title="平均跳数",
                color="tab:orange",
            )
            self._stacked_chart(
                axes[1, 1],
                strategies,
                total=[row.get("tasks", 0) for row in rows],
                accepted=[row.get("accepted", 0) for row in rows],
            )

            plt.tight_layout()
            plot_path = session_dir / f"round_{round_idx}_comparison.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as exc:  # pragma: no cover
            print(f"警告：生成轮次图时出错：{exc}")

    def _plot_summary(self, session_dir: Path) -> None:
        if not HAS_MATPLOTLIB or not self._round_logs:
            return

        try:
            metrics = ("acceptance_rate", "avg_r_t", "avg_hops", "avg_completion_duration")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("多轮次指标趋势", fontsize=16)

            rounds = range(1, len(self._round_logs) + 1)
            strategies = sorted(
                {
                    row.get("strategy", "unknown")
                    for round_data in self._round_logs
                    for row in round_data["results"]
                }
            )

            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2][idx % 2]
                for strategy in strategies:
                    values = []
                    for round_data in self._round_logs:
                        for row in round_data["results"]:
                            if row.get("strategy") == strategy:
                                values.append(row.get(metric, 0.0))
                                break
                        else:
                            values.append(0.0)
                    ax.plot(rounds, values, marker="o", label=strategy)
                ax.set_title(metric)
                ax.set_xlabel("Round")
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            summary_plot = session_dir / "summary_trend.png"
            plt.savefig(summary_plot, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as exc:  # pragma: no cover
            print(f"警告：生成汇总图时出错：{exc}")

    @staticmethod
    def _bar_chart(ax, strategies: List[str], values: List[float], *, title: str, color: str = "tab:blue") -> None:
        ax.bar(strategies, values, color=color, alpha=0.8)
        ax.set_title(title)
        ax.set_xticklabels(strategies, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    @staticmethod
    def _stacked_chart(ax, strategies: List[str], *, total: List[int], accepted: List[int]) -> None:
        ax.bar(strategies, total, label="任务数", color="tab:blue", alpha=0.6)
        ax.bar(strategies, accepted, label="接受数", color="tab:red", alpha=0.6)
        ax.set_title("任务数量对比")
        ax.set_xticklabels(strategies, rotation=30, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    #endregion 绘图

    #region 日志
    def _write_log_header(self) -> None:
        if not self._log_file:
            return
        self._log_file.write("=" * 80 + "\n")
        self._log_file.write("测试输出日志\n")
        if self._session_dir:
            self._log_file.write(f"session_dir: {self._session_dir}\n")
        self._log_file.write(f"start: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        self._log_file.write("=" * 80 + "\n")
        self._log_file.flush()

    #endregion 日志

    #region 详细日志
    def create_detail_logger(self) -> "DetailedTestLogger":
        """
        创建详细测试日志记录器，为单策略小参数测试输出时间步级别日志。
        """

        if self._session_dir:
            detail_dir = Path(self._session_dir)
            inline = True
        else:
            detail_dir = _ensure_dir(self._detail_root)
            inline = False
        return DetailedTestLogger(output_dir=str(detail_dir), inline_files=inline)

    def begin_step_logging(
        self,
        *,
        strategy_name: str,
        model_name: str,
        config: Dict[str, Any],
    ) -> "DetailedTestLogger":
        """
        开启时间步日志记录，返回活跃的 DetailedTestLogger。
        """

        logger = self.create_detail_logger()
        logger.start_test(
            model_name=model_name,
            config=config,
            strategy_name=strategy_name,
        )
        self._active_detail_logger = logger
        return logger

    def log_time_step(
        self,
        *,
        time_step: int,
        current_time: float,
        env: SimuVNEEnv,
        task_info: Optional[Dict[str, Any]] = None,
        placement_result: Optional[Dict[str, Any]] = None,
        completed_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        将单个时间步的信息写入活跃的详细日志。

        说明:
            - 每个时间步都会记录当前时间；
            - 若发生接受/拒绝/完成事件（placement_result 或 completed_tasks），
              会自动捕获完整的 SN 资源状态快照，满足“深入网络映射结构”的要求。
        """

        if not self._active_detail_logger:
            return
        normalized_task = task_info
        if task_info and "vn_nodes_detail" in task_info:
            normalized_task = _denormalize_task_info(task_info, env)
        self._active_detail_logger.log_time_step(
            time_step=time_step,
            current_time=current_time,
            env=env,
            task_info=normalized_task,
            placement_result=placement_result,
            completed_tasks=completed_tasks,
        )

    def end_step_logging(self, *, summary: Dict[str, Any]) -> None:
        """
        结束时间步日志记录，并写入总结。
        """

        if not self._active_detail_logger:
            return
        self._active_detail_logger.finalize(summary=summary)
        self._active_detail_logger = None

    #endregion 详细日志


class DetailedTestLogger:
    """
    详细测试日志记录器，用于记录每次放置的详细信息。
    """

    def __init__(self, output_dir: str, *, inline_files: bool = False) -> None:
        self.output_dir = output_dir
        self.inline_files = inline_files
        self.test_dir: Optional[str] = None
        self.log_file: Optional[Any] = None
        self.time_steps: List[Dict[str, Any]] = []
        self.test_config: Optional[Dict[str, Any]] = None
        self._file_prefix: str = ""

    def start_test(
        self,
        *,
        model_name: str,
        config: Dict[str, Any],
        strategy_name: str,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = f"{strategy_name}_{model_name}_{timestamp}"

        if self.inline_files:
            self.test_dir = self.output_dir
            self._file_prefix = f"{test_name}_"
            os.makedirs(self.test_dir, exist_ok=True)
        else:
            self.test_dir = os.path.join(self.output_dir, test_name)
            os.makedirs(self.test_dir, exist_ok=True)
            self._file_prefix = ""

        self.test_config = {
            "model_name": model_name,
            "strategy_name": strategy_name,
            "timestamp": timestamp,
            "config": config,
        }
        config_path = os.path.join(self.test_dir, f"{self._file_prefix}test_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.test_config, f, indent=2, ensure_ascii=False)

        log_path = os.path.join(self.test_dir, f"{self._file_prefix}detailed_log.txt")
        self.log_file = open(log_path, "w", encoding="utf-8")
        self._write_header()
        self.time_steps = []
        print(f"[详细测试] 输出路径: {self.test_dir}")

    def log_time_step(
        self,
        *,
        time_step: int,
        current_time: float,
        env: SimuVNEEnv,
        task_info: Optional[Dict[str, Any]] = None,
        placement_result: Optional[Dict[str, Any]] = None,
        completed_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        completed_tasks = completed_tasks or []
        has_events = bool(task_info or placement_result or completed_tasks)
        sn_state = self._get_detailed_sn_state(env) if has_events else None
        step_record: Dict[str, Any] = {
            "time_step": time_step,
            "current_time": float(current_time),
            "sn_state": sn_state,
            "active_workflows": len(env.active_workflows),
            "arrived_count": env.arrived_count,
            "accepted_count": env.accepted_count,
            "task_info": task_info,
            "placement_result": placement_result,
            "completed_tasks": completed_tasks,
        }
        self.time_steps.append(step_record)
        if self.log_file:
            self._write_time_step(step_record, minimal=not has_events)
            self.log_file.flush()

    def finalize(self, *, summary: Dict[str, Any]) -> None:
        if not self.test_dir:
            return
        time_steps_path = os.path.join(self.test_dir, f"{self._file_prefix}time_steps.json")
        with open(time_steps_path, "w", encoding="utf-8") as f:
            json.dump(self.time_steps, f, indent=2, ensure_ascii=False)
        summary_path = os.path.join(self.test_dir, f"{self._file_prefix}test_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if self.log_file:
            self.log_file.write(f"\n{'='*80}\n")
            self.log_file.write("测试摘要\n")
            self.log_file.write(f"{'='*80}\n")
            for key, value in summary.items():
                self.log_file.write(f"{key}: {value}\n")
            self.log_file.write(
                f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.log_file.close()
            self.log_file = None
        print(f"[详细测试] 测试结果已保存到: {self.test_dir}")

    def _write_header(self) -> None:
        if not self.log_file:
            return
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write("详细测试日志\n")
        self.log_file.write("=" * 80 + "\n")
        if self.test_config:
            self.log_file.write(f"模型名称: {self.test_config['model_name']}\n")
            self.log_file.write(f"策略名称: {self.test_config['strategy_name']}\n")
            self.log_file.write(f"测试时间: {self.test_config['timestamp']}\n")
            self.log_file.write("\n测试配置:\n")
            for key, value in self.test_config["config"].items():
                self.log_file.write(f"  {key}: {value}\n")
        self.log_file.write("=" * 80 + "\n\n")

    def _write_time_step(
        self, step_record: Dict[str, Any], *, minimal: bool = False
    ) -> None:
        if not self.log_file:
            return
        self.log_file.write(f"\n{'='*80}\n")
        self.log_file.write(
            f"时间步 {step_record['time_step']} (t={step_record['current_time']:.2f})\n"
        )
        if minimal:
            return
        sn_state = step_record["sn_state"]
        if sn_state:
            self.log_file.write(f"\n【SN网络状态】\n")
            self.log_file.write(
                f"  节点数: {sn_state['total_nodes']}, 边数: {sn_state['total_edges']}\n"
            )
            self.log_file.write(
                f"  活跃工作流: {step_record['active_workflows']}\n"
                f"  已到达任务: {step_record['arrived_count']}, "
                f"已接受任务: {step_record['accepted_count']}\n"
            )
            self.log_file.write(f"\n【SN节点资源详情】\n")
            for node in sn_state["nodes"]:
                self.log_file.write(
                    f"  节点{node['node_id']}: "
                    f"CPU({node['cpu_res']:.2f}/{node['cpu_init']:.2f}), "
                    f"MEM({node['mem_res']:.2f}/{node['mem_init']:.2f}), "
                    f"DISK({node['disk_res']:.2f}/{node['disk_init']:.2f})\n"
                )
        if step_record["completed_tasks"]:
            self.log_file.write(f"\n【任务完成】\n")
            for completed in step_record["completed_tasks"]:
                completion_time = completed.get("completion_time")
                completion_str = (
                    f"{completion_time:.2f}" if completion_time is not None else "N/A"
                )
                self.log_file.write(
                    f"  任务ID {completed.get('task_id', 'N/A')} 于 "
                    f"{completion_str} 释放资源\n"
                )
        if step_record["task_info"]:
            task_info = step_record["task_info"]
            self.log_file.write(f"\n【任务到达】\n")
            self.log_file.write(f"  任务ID: {task_info.get('task_id', 'N/A')}\n")
            self.log_file.write(f"  工作流类型: {task_info.get('workflow', 'N/A')}\n")
            self.log_file.write(f"  VN节点数: {task_info.get('vn_nodes', 'N/A')}\n")
            lifetime = task_info.get("lifetime")
            if lifetime is not None:
                self.log_file.write(f"  生存时间: {float(lifetime):.2f}\n")
            if "vn_nodes_detail" in task_info:
                self.log_file.write("  VN节点详情:\n")
                for vn_node in task_info["vn_nodes_detail"]:
                    self.log_file.write(
                        f"    节点{vn_node['idx']}: "
                        f"CPU={vn_node['cpu']:.4f}, "
                        f"MEM={vn_node['mem']:.4f}, "
                        f"DISK={vn_node['disk']:.4f}\n"
                    )
        placement = step_record["placement_result"]
        if placement:
            success = placement.get("success", False)
            self.log_file.write(f"\n【放置结果】\n")
            self.log_file.write(f"  成功: {success}\n")
            self.log_file.write(
                f"  放置时间: {placement.get('placement_time', 0.0)*1000:.2f}ms\n"
            )
            mapping = placement.get("mapping", {})
            if mapping:
                label = "映射关系" if success else "映射关系（部分）"
                self.log_file.write(f"  {label}:\n")
                for vn_idx, sn_id in sorted(mapping.items()):
                    self.log_file.write(f"    VN节点{vn_idx} → SN节点{sn_id}\n")
            meta = placement.get("metadata", {})
            if meta:
                self.log_file.write("  策略元数据:\n")
                for key, value in meta.items():
                    if key in ("failure_debug", "success_debug"):
                        continue
                    self.log_file.write(f"    {key}: {value}\n")
                if meta.get("success_debug"):
                    self._write_probs_matrix(meta["success_debug"], is_success=True)
                elif meta.get("failure_debug"):
                    self._write_failure_debug(meta["failure_debug"])
                elif "probs_matrix" in meta:
                    debug_info = {
                        "probs_matrix": meta.get("probs_matrix"),
                        "final_mapping": mapping if success else {},
                    }
                    self._write_probs_matrix(debug_info, is_success=success)
            if success and placement.get("paths"):
                paths = placement["paths"]
                self.log_file.write(f"  路径数: {len(paths)}\n")
                if paths:
                    total_hops = sum(len(p) - 1 for p in paths)
                    avg_hops = total_hops / len(paths) if paths else 0.0
                    self.log_file.write(f"  平均跳数: {avg_hops:.2f}\n")
            elif not success:
                total_vn_nodes = placement.get("total_vn_nodes")
                mapped_vn_nodes = placement.get("mapped_vn_nodes")
                if total_vn_nodes is not None and mapped_vn_nodes is not None:
                    self.log_file.write(
                        f"  已映射/需求节点: {mapped_vn_nodes}/{total_vn_nodes}\n"
                    )
                failure_reason = placement.get("failure_reason", "无法完成映射")
                self.log_file.write(f"  失败原因: {failure_reason}\n")

    def _write_probs_matrix(self, debug_info: Dict[str, Any], is_success: bool = True) -> None:
        if not self.log_file:
            return
        label = "[成功调试信息]" if is_success else "[失败调试信息]"
        self.log_file.write(f"    {label}\n")
        probs = debug_info.get("probs_matrix")
        if probs:
            self.log_file.write("      概率矩阵:\n")
            for idx, row in enumerate(probs):
                row_str = ", ".join(f"{float(val):.6f}" for val in row)
                self.log_file.write(f"        VN{idx}: [{row_str}]\n")
        if is_success:
            final_mapping = debug_info.get("final_mapping", {})
            if final_mapping:
                self.log_file.write("      最终映射关系:\n")
                for vn_idx, sn_id in sorted(final_mapping.items()):
                    self.log_file.write(f"        VN节点{vn_idx} → SN节点{sn_id}\n")
        snapshot = debug_info.get("sn_snapshot")
        if snapshot:
            label_sn = "放置后SN资源" if is_success else "回滚前SN资源"
            self.log_file.write(f"      {label_sn}:\n")
            for node in snapshot:
                self.log_file.write(
                    f"        SN{node.get('sn_node', 'N/A')}: "
                    f"CPU={node.get('cpu_res', 0.0):.4f}, "
                    f"MEM={node.get('mem_res', 0.0):.4f}, "
                    f"DISK={node.get('disk_res', 0.0):.4f}\n"
                )

    def _write_failure_debug(self, failure_debug: Dict[str, Any]) -> None:
        if not self.log_file:
            return
        self.log_file.write("    [失败调试信息]\n")
        failed_vn = failure_debug.get("failed_vn")
        if failed_vn is not None:
            self.log_file.write(f"      失败VN节点: {failed_vn}\n")
        if "reason" in failure_debug:
            self.log_file.write(f"      失败原因: {failure_debug['reason']}\n")
        self._write_probs_matrix(failure_debug, is_success=False)

    def _get_detailed_sn_state(self, env: SimuVNEEnv) -> Dict[str, Any]:
        node_list = sorted(env.G_sn.nodes())
        nodes = []
        for node_id in node_list:
            node_data = env.G_sn.nodes[node_id]
            initial = env._sn_initial_state.get(node_id, {})
            nodes.append(
                {
                    "node_id": node_id,
                    "cpu_res": float(node_data.get("cpu_res", 0.0)),
                    "mem_res": float(node_data.get("mem_res", 0.0)),
                    "disk_res": float(node_data.get("disk_res", 0.0)),
                    "cpu_init": float(initial.get("cpu", 0.0)),
                    "mem_init": float(initial.get("mem", 0.0)),
                    "disk_init": float(initial.get("disk", 0.0)),
                    "cpu_util": 1.0
                    - (node_data.get("cpu_res", 0.0) / (initial.get("cpu", 1.0) + 1e-8)),
                    "mem_util": 1.0
                    - (node_data.get("mem_res", 0.0) / (initial.get("mem", 1.0) + 1e-8)),
                    "disk_util": 1.0
                    - (node_data.get("disk_res", 0.0) / (initial.get("disk", 1.0) + 1e-8)),
                }
            )
        edges = []
        for u, v, data in env.G_sn.edges(data=True):
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "bandwidth": float(data.get("bandwidth", 0.0)),
                }
            )
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }



