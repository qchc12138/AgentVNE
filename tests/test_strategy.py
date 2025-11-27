from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

import numpy as np
from torch_geometric.data import Data

import os
import sys
from pathlib import Path

#region sys.path 管理
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env import SimuVNEEnv, WorkflowGenerator

try:  # 兼容后续还未实现 TestPrinter 的阶段
    from tests.test_printer import TestPrinter, ResourceFailureTracker  # type: ignore
except Exception:  # pragma: no cover - 在初期阶段允许缺失
    TestPrinter = None  # type: ignore

    class ResourceFailureTracker:  # type: ignore
        def __init__(self) -> None:
            self._records: List[Dict[str, Any]] = []

        def record_failure(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            entry: Dict[str, Any] = {}
            self._records.append(entry)
            return entry

        def build_summary(self) -> Dict[str, Any]:
            return {"records": list(self._records), "max_resource_failure_ratio": 0.0}

__all__ = [
    "StrategyContext",
    "StrategyResult",
    "PlacementStrategy",
    "TestConfig",
    "SingleTester",
    "run_single_strategy_test",
    "format_config_info",
    "run_strategy_with_details",
    "build_strategy_row",
]


#region 数据结构
@dataclass
class StrategyContext:
    """策略在一次放置决策中共享的上下文。"""

    rng: np.random.Generator
    step_id: int
    verbose: bool = False


@dataclass
class StrategyResult:
    """策略统一输出格式。"""

    success: bool
    mapping: Dict[int, int]
    metadata: Dict[str, Any]


@runtime_checkable
class PlacementStrategy(Protocol):
    """
    所有策略需实现的最小接口。

    注意：策略实现只能读取 env/sn_state，不得直接修改 env 内部资源，
    资源扣减、任务登记等副作用统一由测试框架负责。
    """

    name: str

    def prepare(self, env: SimuVNEEnv) -> None:  # noqa: D401
        """策略执行前的准备逻辑。"""

    def place(
        self,
        vn: Data,
        sn_state: Data,
        env: SimuVNEEnv,
        *,
        context: StrategyContext,
    ) -> StrategyResult:
        """对单个 VN 任务执行放置决策。"""


@dataclass
class TestConfig:
    """单策略测试配置。"""

    sn_topology_path: str
    workflow_types: Dict[str, str]
    arrival_rate: float = 1.0
    mean_lifetime: float = 10.0
    max_time_steps: int = 1000
    device: str = "cpu"
    seed: int = 0
    penalty: float = -150.0


@dataclass
class _TesterRuntime:
    env: SimuVNEEnv
    wf_gen: WorkflowGenerator
    strategy: PlacementStrategy
    rng: np.random.Generator


#endregion 数据结构


#region 核心执行单元
class SingleTester:
    """
    单策略单配置测试的最小执行单元。

    tester.py 负责 orchestrate，本类负责“准备 + 运行”。
    """

    def __init__(self, config: TestConfig):
        self.config = dataclasses.replace(config)
        expectation = max(self.config.arrival_rate * self.config.max_time_steps, 0.0)
        self._task_limit = max(int(expectation * 2 + 10), 10)
        self._runtime: Optional[_TesterRuntime] = None

    #region 对外接口
    def prepare(self, strategy: PlacementStrategy) -> None:
        """基于配置初始化环境与策略。"""

        cfg = self.config
        env = SimuVNEEnv(
            sn_topology_path=cfg.sn_topology_path,
            device=cfg.device,
            penalty=cfg.penalty,
            max_arrived_tasks=self._task_limit,
        )
        env.reset()

        sn_capacity = env.get_sn_max_capacity()
        wf_gen = WorkflowGenerator(
            workflow_types=cfg.workflow_types,
            arrival_rate=cfg.arrival_rate,
            mean_lifetime=cfg.mean_lifetime,
            seed=cfg.seed,
            sn_capacity_for_norm=sn_capacity,
        )

        rng = np.random.default_rng(cfg.seed)
        strategy.prepare(env)

        self._runtime = _TesterRuntime(
            env=env,
            wf_gen=wf_gen,
            strategy=strategy,
            rng=rng,
        )

    def run(self, *, show_details: bool = False) -> Dict[str, Any]:
        """执行一次测试流程。"""

        if self._runtime is None:
            raise RuntimeError("请先调用 prepare(strategy) 初始化测试环境。")

        runtime = self._runtime
        env = runtime.env
        wf_gen = runtime.wf_gen
        cfg = self.config

        tasks: List[Dict[str, Any]] = []
        failure_tracker = ResourceFailureTracker()
        time_step = 0
        stop_arrival_time: Optional[float] = None

        if show_details:
            print(
                f"[SingleTester] start: rate={cfg.arrival_rate}, "
                f"lifetime={cfg.mean_lifetime}, steps={cfg.max_time_steps}"
            )

        while (
            time_step < cfg.max_time_steps
            and not env.is_done()
            and env.arrived_count < self._task_limit
        ):
            env.step_time(1.0)
            if wf_gen.check_arrival(1.0) and not env.is_done():
                wf_type = wf_gen.sample_workflow_type()
                self._dispatch_single_task(
                    runtime=runtime,
                    wf_name=wf_type,
                    tasks=tasks,
                    time_step=time_step,
                    show_details=show_details,
                    failure_tracker=failure_tracker,
                )
            time_step += 1

        stop_arrival_time = float(env.current_time)
        limit_reached = env.arrived_count >= self._task_limit

        while env.active_workflows:
            env.step_time(1.0)
            time_step += 1

        end_time = float(env.current_time)
        lag_time = max(0.0, end_time - stop_arrival_time)
        summary = _compute_task_summary(tasks, failure_tracker=failure_tracker)

        if limit_reached:
            print(
                "[SingleTester] ⚠️ 触发内部任务数量上限，"
                "请调整 arrival_rate 或 max_time_steps。"
            )

        if show_details:
            print(
                f"[SingleTester] done: tasks={summary['total_tasks']}, "
                f"accepted={summary['accepted']}, "
                f"accept_rate={summary['acceptance_rate']*100:.2f}%"
            )

        return {
            "tasks": tasks,
            "summary": summary,
            "lag_time": lag_time,
            "stop_arrival_time": stop_arrival_time,
            "end_time": end_time,
        }

    #endregion 对外接口

    #region 内部方法
    def _dispatch_single_task(
        self,
        *,
        runtime: _TesterRuntime,
        wf_name: str,
        tasks: List[Dict[str, Any]],
        time_step: int,
        show_details: bool,
        failure_tracker: Optional["ResourceFailureTracker"] = None,
    ) -> None:
        """生成并调度单个 workflow 任务。"""

        env = runtime.env
        wf_gen = runtime.wf_gen
        strategy = runtime.strategy
        rng = runtime.rng

        lifetime = wf_gen.sample_lifetime()
        vn = wf_gen.load_workflow_graph(wf_name)

        task_id = env.arrived_count
        env.arrived_count += 1

        if show_details:
            print(
                f"[t={env.current_time:.1f}] 任务#{task_id} 到达 "
                f"(workflow={wf_name}, nodes={vn.x.size(0)}, life={lifetime:.1f})"
            )

        sn_state = env.get_sn_state()
        context = StrategyContext(rng=rng, step_id=time_step, verbose=show_details)

        start = time.perf_counter()
        result = strategy.place(vn, sn_state, env, context=context)
        placement_time = time.perf_counter() - start

        record: Dict[str, Any] = {
            "task_id": task_id,
            "workflow": wf_name,
            "vn_nodes": int(vn.x.size(0)),
            "lifetime": float(lifetime),
            "arrival_time": float(env.current_time),
            "placement_time": float(placement_time),
            "success": bool(result.success),
            "metadata": dict(result.metadata or {}),
            "completion_time": None,
            "completion_duration": None,
            "r_t": None,
            "hops": None,
        }

        if result.success:
            vn_paths = env._compute_paths_and_bw_demand(vn, result.mapping)
            if vn_paths is None:
                record["success"] = False
                record["failure_reason"] = "路径不存在"
            else:
                try:
                    env._apply_mapping(vn, result.mapping, vn_paths)
                except ValueError as exc:
                    record["success"] = False
                    record["failure_reason"] = f"资源扣减失败: {exc}"
                if record["success"]:
                    expire_time = env.current_time + float(lifetime)
                    env.active_workflows.append(
                        {
                            "vn": vn,
                            "mapping": result.mapping,
                            "paths": vn_paths,
                            "expire_time": expire_time,
                            "task_id": task_id,
                        }
                    )
                    env.accepted_count += 1
                    record["completion_time"] = float(expire_time)
                    record["completion_duration"] = float(
                        record["completion_time"] - record["arrival_time"]
                    )
                    record["hops"] = float(
                        sum(max(len(path) - 1, 0) for _, _, path in vn_paths if path is not None)
                    )
                    record["r_t"] = float(env._compute_rt())
                    if show_details:
                        print(
                            f"  ✓ 放置成功: complete@{record['completion_time']:.1f}, "
                            f"hops={record['hops']:.0f}, r_t={record['r_t']:.3f}"
                        )

        if not record["success"]:
            record.setdefault("failure_reason", "策略拒绝或资源不足")
            if failure_tracker is not None:
                stats = failure_tracker.record_failure(
                    env=env,
                    vn=vn,
                    task_id=task_id,
                    failure_reason=record.get("failure_reason"),
                )
                record["resource_failure_stats"] = stats
            if show_details:
                print(f"  ✗ 放置失败: {record['failure_reason']}")

        tasks.append(record)

    #endregion 内部方法


#endregion 核心执行单元


#region 辅助方法
def _compute_task_summary(
    tasks: List[Dict[str, Any]],
    *,
    failure_tracker: Optional["ResourceFailureTracker"] = None,
) -> Dict[str, Any]:
    """对任务记录进行统计。"""

    failure_summary = (
        failure_tracker.build_summary()
        if failure_tracker is not None
        else {"records": [], "max_resource_failure_ratio": 0.0}
    )

    if not tasks:
        return {
            "acceptance_rate": 0.0,
            "total_tasks": 0,
            "accepted": 0,
            "avg_r_t": 0.0,
            "avg_hops": 0.0,
            "max_hops": 0.0,
            "avg_completion_duration": 0.0,
            "resource_failure_records": failure_summary["records"],
            "max_resource_failure_ratio": failure_summary["max_resource_failure_ratio"],
        }

    total = len(tasks)
    accepted_tasks = [t for t in tasks if t["success"]]
    accepted = len(accepted_tasks)

    r_ts = [t["r_t"] for t in accepted_tasks if t.get("r_t") is not None]
    hops = [t["hops"] for t in accepted_tasks if t.get("hops") is not None]
    completion = [
        t["completion_duration"] for t in accepted_tasks if t.get("completion_duration") is not None
    ]

    return {
        "total_tasks": total,
        "accepted": accepted,
        "acceptance_rate": accepted / total if total else 0.0,
        "avg_r_t": float(np.mean(r_ts)) if r_ts else 0.0,
        "avg_hops": float(np.mean(hops)) if hops else 0.0,
        "max_hops": float(max(hops)) if hops else 0.0,
        "avg_completion_duration": float(np.mean(completion)) if completion else 0.0,
        "resource_failure_records": failure_summary["records"],
        "max_resource_failure_ratio": failure_summary["max_resource_failure_ratio"],
    }


def _build_strategy_row(strategy_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """构造用于日志输出的策略结果摘要。"""

    summary = result["summary"]
    return {
        "strategy": strategy_name,
        "acceptance_rate": summary["acceptance_rate"] * 100.0,
        "avg_r_t": summary.get("avg_r_t", 0.0),
        "avg_hops": summary.get("avg_hops", 0.0),
        "max_hops": summary.get("max_hops", 0.0),
        "avg_completion_duration": summary.get("avg_completion_duration", 0.0),
        "tasks": summary["total_tasks"],
        "accepted": summary["accepted"],
        "lag_time": result.get("lag_time", 0.0),
        "stop_arrival_time": result.get("stop_arrival_time", 0.0),
        "end_time": result.get("end_time", 0.0),
        "max_resource_failure_ratio": summary.get("max_resource_failure_ratio", 0.0),
    }


def build_strategy_row(strategy_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """对外暴露的策略结果摘要构造函数。"""

    return _build_strategy_row(strategy_name, result)


def format_config_info(
    config: TestConfig,
    *,
    workflow_keys: Iterable[str],
) -> Dict[str, Any]:
    """格式化配置，便于 TestPrinter 打印。"""

    return {
        "sn_topology": config.sn_topology_path,
        "workflow_types": ", ".join(sorted(workflow_keys)),
        "arrival_rate": config.arrival_rate,
        "mean_lifetime": config.mean_lifetime,
        "max_time_steps": config.max_time_steps,
        "device": config.device,
        "seed": config.seed,
        "penalty": config.penalty,
    }


def _detail_logger_config(config: TestConfig) -> Dict[str, Any]:
    return {
        "sn_topology_path": config.sn_topology_path,
        "workflow_types": ", ".join(sorted(config.workflow_types.keys())),
        "arrival_rate": config.arrival_rate,
        "mean_lifetime": config.mean_lifetime,
        "max_time_steps": config.max_time_steps,
        "device": config.device,
        "seed": config.seed,
        "penalty": config.penalty,
    }


def _collect_completed_tasks(env: SimuVNEEnv, cutoff_time: float) -> List[Dict[str, float]]:
    """
    仅收集将在 cutoff_time 前到期的 workflow 信息，交由 env.step_time 负责实际释放。
    """
    completed: List[Dict[str, float]] = []
    for wf in env.active_workflows:
        expire_time = float(wf.get("expire_time", float("inf")))
        if expire_time <= cutoff_time + 1e-9:
            completed.append(
                {
                    "task_id": wf.get("task_id"),
                    "completion_time": expire_time,
                }
            )
    return completed


def _update_completed_records(
    completed_infos: List[Dict[str, float]],
    task_records: Dict[int, Dict[str, Any]],
    env: SimuVNEEnv,
) -> None:
    for completed in completed_infos:
        task_id = completed.get("task_id")
        if task_id is None:
            continue
        record = task_records.get(task_id)
        if record is None:
            continue
        completion_time_value = float(completed.get("completion_time", env.current_time))
        record["completion_time"] = completion_time_value
        arrival_time = record.get("arrival_time")
        if arrival_time is not None:
            record["completion_duration"] = max(
                0.0, completion_time_value - float(arrival_time)
            )


def _build_task_info(
    *,
    task_id: int,
    workflow_name: str,
    vn: Data,
    lifetime: float,
) -> Dict[str, Any]:
    vn_nodes_detail = []
    for i in range(vn.x.size(0)):
        feats = vn.x[i]
        vn_nodes_detail.append(
            {
                "idx": i,
                "cpu": float(feats[0].item()),
                "mem": float(feats[1].item()),
                "disk": float(feats[2].item()),
            }
        )
    return {
        "task_id": task_id,
        "workflow": workflow_name,
        "vn_nodes": int(vn.x.size(0)),
        "lifetime": float(lifetime),
        "vn_nodes_detail": vn_nodes_detail,
    }


StrategyFactory = Union[PlacementStrategy, Callable[[], PlacementStrategy]]
TesterType = Type[SingleTester]


def run_single_strategy_test(
    *,
    strategy_factory: StrategyFactory,
    tester_cls: TesterType = SingleTester,
    detail_print: bool = False,
    config: Optional[TestConfig] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    printer: Optional["TestPrinter"] = None,
    strategy_label: Optional[str] = None,
) -> Dict[str, Any]:
    """统一封装单策略测试入口。"""

    if config is None:
        if not config_overrides:
            raise ValueError("必须提供测试配置（config 或 config_overrides）。")
        config = TestConfig(**config_overrides)
    elif config_overrides:
        config = dataclasses.replace(config, **config_overrides)

    strategy = strategy_factory() if callable(strategy_factory) else strategy_factory
    tester = tester_cls(config)
    tester.prepare(strategy)
    result = tester.run(show_details=detail_print)

    if printer is not None:
        label = strategy_label or getattr(strategy, "name", strategy.__class__.__name__)
        printer.add_row(_build_strategy_row(label, result))

    return result


#endregion 辅助方法


def run_strategy_with_details(
    *,
    strategy_factory: StrategyFactory,
    config: TestConfig,
    printer: "TestPrinter",
    detail_print: bool = False,
) -> Dict[str, Any]:
    """
    执行单策略测试并输出逐时间步详细日志。
    """

    strategy = strategy_factory() if callable(strategy_factory) else strategy_factory
    task_limit = max(int(max(config.arrival_rate, 0.0) * config.max_time_steps * 2 + 10), 10)

    env = SimuVNEEnv(
        sn_topology_path=config.sn_topology_path,
        device=config.device,
        penalty=config.penalty,
        max_arrived_tasks=task_limit,
    )
    env.reset()
    sn_capacity = env.get_sn_max_capacity()
    wf_gen = WorkflowGenerator(
        workflow_types=config.workflow_types,
        arrival_rate=config.arrival_rate,
        mean_lifetime=config.mean_lifetime,
        seed=config.seed,
        sn_capacity_for_norm=sn_capacity,
    )
    rng = np.random.default_rng(config.seed)
    strategy.prepare(env)

    printer.begin_step_logging(
        strategy_name=getattr(strategy, "name", strategy.__class__.__name__),
        model_name=getattr(strategy, "model_name", getattr(strategy, "name", strategy.__class__.__name__)),
        config=_detail_logger_config(config),
    )

    tasks: List[Dict[str, Any]] = []
    task_records: Dict[int, Dict[str, Any]] = {}
    failure_tracker = ResourceFailureTracker()
    time_step = 0
    time_delta = 1.0
    stop_arrival_time: Optional[float] = None

    while (
        time_step < config.max_time_steps
        and not env.is_done()
    ):
        completion_time = env.current_time + time_delta
        completed_task_infos = _collect_completed_tasks(env, completion_time)

        env.step_time(time_delta=time_delta)
        _update_completed_records(completed_task_infos, task_records, env)

        has_arrival = wf_gen.check_arrival(time_unit=1.0)
        task_info = None
        placement_result = None

        if has_arrival and not env.is_done():
            wf_type = wf_gen.sample_workflow_type()
            vn = wf_gen.load_workflow_graph(wf_type)
            lifetime = wf_gen.sample_lifetime()
            task_id = env.arrived_count
            env.arrived_count += 1

            task_info = _build_task_info(
                task_id=task_id,
                workflow_name=wf_type,
                vn=vn,
                lifetime=lifetime,
            )

            sn_state = env.get_sn_state()
            context = StrategyContext(rng=rng, step_id=time_step, verbose=detail_print)
            start_time = time.perf_counter()
            place_result = strategy.place(vn, sn_state, env, context=context)
            placement_time = time.perf_counter() - start_time

            placement_result = {
                "success": place_result.success,
                "mapping": dict(place_result.mapping),
                "metadata": dict(place_result.metadata or {}),
                "placement_time": placement_time,
                "total_vn_nodes": int(vn.x.size(0)),
                "mapped_vn_nodes": len(place_result.mapping),
            }

            record = {
                "task_id": task_id,
                "workflow": wf_type,
                "vn_nodes": int(vn.x.size(0)),
                "lifetime": float(lifetime),
                "arrival_time": float(env.current_time),
                "placement_time": float(placement_time),
                "success": bool(place_result.success),
                "metadata": dict(place_result.metadata or {}),
                "r_t": None,
                "hops": None,
                "completion_time": None,
                "completion_duration": None,
            }

            if place_result.success:
                vn_paths = env._compute_paths_and_bw_demand(vn, place_result.mapping)
                if vn_paths is None:
                    placement_result["success"] = False
                    placement_result["failure_reason"] = "路径不存在"
                    record["success"] = False
                    record["failure_reason"] = "路径不存在"
                else:
                    try:
                        env._apply_mapping(vn, place_result.mapping, vn_paths)
                    except ValueError as exc:
                        failure_msg = f"资源扣减失败: {exc}"
                        placement_result["success"] = False
                        placement_result["failure_reason"] = failure_msg
                        record["success"] = False
                        record["failure_reason"] = failure_msg
                    if record["success"]:
                        expire_time = env.current_time + float(lifetime)
                        env.active_workflows.append(
                            {
                                "vn": vn,
                                "mapping": place_result.mapping,
                                "paths": vn_paths,
                                "expire_time": expire_time,
                                "task_id": task_id,
                            }
                        )
                        env.accepted_count += 1
                        record["completion_time"] = float(expire_time)
                        record["completion_duration"] = float(
                            record["completion_time"] - record["arrival_time"]
                        )
                        record["hops"] = float(
                            sum(max(len(path) - 1, 0) for _, _, path in vn_paths if path is not None)
                        )
                        record["r_t"] = float(env._compute_rt())
                        placement_result["paths"] = vn_paths
            else:
                failure_reason = place_result.metadata.get("failure_reason", "无法完成映射")
                placement_result["failure_reason"] = failure_reason
                record["failure_reason"] = failure_reason

            if not record["success"]:
                stats = failure_tracker.record_failure(
                    env=env,
                    vn=vn,
                    task_id=task_id,
                    failure_reason=record.get("failure_reason"),
                )
                record["resource_failure_stats"] = stats
                if placement_result is not None:
                    placement_result.setdefault("failure_reason", record["failure_reason"])
                    placement_result["resource_failure_stats"] = stats

            tasks.append(record)
            task_records[task_id] = record

        printer.log_time_step(
            time_step=time_step,
            current_time=env.current_time,
            env=env,
            task_info=task_info,
            placement_result=placement_result,
            completed_tasks=completed_task_infos,
        )
        time_step += 1

    stop_arrival_time = float(env.current_time)

    while env.active_workflows:
        completion_time = env.current_time + time_delta
        completed_task_infos = _collect_completed_tasks(env, completion_time)
        env.step_time(time_delta=time_delta)
        _update_completed_records(completed_task_infos, task_records, env)
        printer.log_time_step(
            time_step=time_step,
            current_time=env.current_time,
            env=env,
            completed_tasks=completed_task_infos,
        )
        time_step += 1

    end_time = float(env.current_time)
    lag_time = max(0.0, end_time - stop_arrival_time)
    summary = _compute_task_summary(tasks, failure_tracker=failure_tracker)
    printer.end_step_logging(summary=summary)

    return {
        "tasks": tasks,
        "summary": summary,
        "lag_time": lag_time,
        "stop_arrival_time": stop_arrival_time,
        "end_time": end_time,
    }

