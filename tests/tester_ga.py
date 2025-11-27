from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Type

import os
import sys
from pathlib import Path

from torch_geometric.data import Data

# region sys.path 管理
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# endregion

from env import SimuVNEEnv
from baselines.genetic_algorithm.ga_config import (
    DEFAULT_GA_OUTPUT_DIR,
    GAParams,
    find_latest_ga_config,
    load_ga_config,
)
from baselines.genetic_algorithm.ga_core import GeneticAlgorithm
from tests.test_configs import get_smoke_config
from tests.test_printer import TestPrinter
from tests.test_strategy import (
    PlacementStrategy,
    SingleTester,
    StrategyContext,
    StrategyResult,
    TestConfig,
    build_strategy_row,
    format_config_info,
    run_single_strategy_test,
    run_strategy_with_details,
)

__all__ = [
    "GAPlacementStrategy",
    "ga_strategy_factory",
    "run_ga_strategy_test",
    "smoke_test_ga_strategy",
]


class GAPlacementStrategy(PlacementStrategy):
    """遗传算法策略包装器：仅读取主环境，在副本上完成搜索。"""

    name: str = "ga"

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        use_latest: bool = True,
        ga_output_dir: str = DEFAULT_GA_OUTPUT_DIR,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 5,
        tournament_size: int = 3,
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.use_latest = use_latest
        self.ga_output_dir = ga_output_dir
        self.verbose = verbose

        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        self._config_source: Optional[str] = None
        self._load_params_if_needed()

    # region 初始化辅助
    def _load_params_if_needed(self) -> None:
        candidate: Optional[str] = None
        if self.config_path:
            candidate = self.config_path
        elif self.use_latest:
            candidate = find_latest_ga_config(self.ga_output_dir)

        if not candidate:
            return

        try:
            params = load_ga_config(candidate, verbose=self.verbose)
        except (FileNotFoundError, ValueError) as exc:
            if self.verbose:
                print(f"[GA] ⚠️ 加载配置失败: {exc}. 使用默认参数。")
            return

        self._apply_params(params)
        self._config_source = candidate

    def _apply_params(self, params: GAParams) -> None:
        self.population_size = params.population_size
        self.max_generations = params.max_generations
        self.crossover_rate = params.crossover_rate
        self.mutation_rate = params.mutation_rate
        self.elite_size = params.elite_size
        self.tournament_size = params.tournament_size
    # endregion

    def prepare(self, env: SimuVNEEnv) -> None:  # noqa: D401, ARG002
        """GA 策略无需额外准备。"""

    def place(
        self,
        vn: Data,
        sn_state: Data,  # noqa: ARG002
        env: SimuVNEEnv,
        *,
        context: StrategyContext,
    ) -> StrategyResult:
        temp_env = copy.deepcopy(env)
        sn_graph = temp_env.G_sn.to_undirected() if temp_env.G_sn.is_directed() else temp_env.G_sn
        sn_node_list = sorted(temp_env.G_sn.nodes())
        sn_resources: Dict[int, Dict[str, float]] = {}
        for sn_id in sn_node_list:
            node = temp_env.G_sn.nodes[sn_id]
            sn_resources[sn_id] = {
                "cpu_res": float(node.get("cpu_res", 0.0)),
                "mem_res": float(node.get("mem_res", 0.0)),
                "disk_res": float(node.get("disk_res", 0.0)),
            }

        ga = GeneticAlgorithm(
            vn=vn,
            sn_graph=sn_graph,
            sn_node_list=sn_node_list,
            sn_max_capacity=temp_env.get_sn_max_capacity(),
            population_size=self.population_size,
            max_generations=self.max_generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            elite_size=self.elite_size,
            tournament_size=self.tournament_size,
            seed=int(context.rng.integers(0, 2**31 - 1)) if context.rng else None,
            verbose=self.verbose or context.verbose,
        )

        mapping, fitness = ga.evolve(sn_resources)
        success = len(mapping) == vn.x.size(0)
        path_length = float(ga.compute_path_length(mapping)) if success else 0.0

        metadata: Dict[str, Any] = {
            "fitness": float(fitness),
            "path_length": path_length,
            "config_source": self._config_source,
            "strategy": self.name,
        }
        if not success:
            metadata["failure_reason"] = "GA 进化后映射不完整"

        if context.verbose:
            msg = "✓" if success else "✗"
            print(f"[GA] {msg} step={context.step_id}, vn_nodes={vn.x.size(0)}")

        return StrategyResult(success=success, mapping=mapping if success else {}, metadata=metadata)


def ga_strategy_factory(
    *,
    config_path: Optional[str] = None,
    use_latest: bool = True,
    ga_output_dir: str = DEFAULT_GA_OUTPUT_DIR,
    population_size: int = 50,
    max_generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 5,
    tournament_size: int = 3,
    verbose: bool = False,
) -> PlacementStrategy:
    """工厂方法，方便 tester.py 动态构建 GA 策略。"""

    return GAPlacementStrategy(
        config_path=config_path,
        use_latest=use_latest,
        ga_output_dir=ga_output_dir,
        population_size=population_size,
        max_generations=max_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        tournament_size=tournament_size,
        verbose=verbose,
    )


def run_ga_strategy_test(
    *,
    detail_print: bool = False,
    tester_cls: Type[SingleTester] = SingleTester,
    config: Optional[TestConfig] = None,
    config_overrides: Optional[Dict[str, object]] = None,
    printer: Optional[TestPrinter] = None,
    strategy_label: str = "ga",
) -> Dict[str, Any]:
    """兼容 run_single_strategy_test 的 GA 策略封装。"""

    return run_single_strategy_test(
        strategy_factory=ga_strategy_factory,
        tester_cls=tester_cls,
        detail_print=detail_print,
        config=config,
        config_overrides=config_overrides,
        printer=printer,
        strategy_label=strategy_label,
    )


def smoke_test_ga_strategy(*, detail_print: bool = False) -> Dict[str, Any]:
    """GA 的小参数冒烟测试。"""

    config = get_smoke_config()
    printer = TestPrinter(
        enable_logging=True,
        enable_plotting=False,
        test_scope="ga_single",
    )
    try:
        printer.start_round(
            table_title="GA Strategy Smoke Test",
            config_info=format_config_info(
                config,
                workflow_keys=config.workflow_types.keys(),
            ),
        )
        result = run_strategy_with_details(
            strategy_factory=ga_strategy_factory,
            config=config,
            printer=printer,
            detail_print=detail_print,
        )
        printer.add_row(
            build_strategy_row("ga", result),
            strategy_info={"strategy_name": "GAPlacementStrategy"},
        )
        printer.finalize()
    finally:
        printer.close()

    return result


if __name__ == "__main__":
    summary = smoke_test_ga_strategy(detail_print=False)["summary"]
    print(
        "[GASmoke] total_tasks={total}, accepted={accepted}, acceptance_rate={rate:.2f}%".format(
            total=summary["total_tasks"],
            accepted=summary["accepted"],
            rate=summary["acceptance_rate"] * 100.0,
        )
    )


