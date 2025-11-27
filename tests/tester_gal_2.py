from __future__ import annotations

import copy
from typing import Dict, Optional, Type

import os
import sys
from pathlib import Path

from torch_geometric.data import Data

#region sys.path 管理
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
#endregion

from env import SimuVNEEnv
from tests.test_printer import TestPrinter
from tests.test_configs import get_smoke_config
from tests.test_strategy import (
    PlacementStrategy,
    SingleTester,
    StrategyContext,
    StrategyResult,
    TestConfig,
    format_config_info,
    run_strategy_with_details,
    build_strategy_row,
)
from baselines.GAL_2 import GreedyAllocator as GAL2Allocator

__all__ = [
    "GAL2PlacementStrategy",
    "run_gal2_strategy_test",
    "smoke_test_gal2_strategy",
]


class GAL2PlacementStrategy(PlacementStrategy):
    """GAL-PN (Per-Node) 策略包装器：逐节点贪心，每次选择后立即扣减资源。"""

    name: str = "gal_pn"

    def prepare(self, env: SimuVNEEnv) -> None:  # noqa: D401, ARG002
        """GAL_2 策略无需额外准备。"""

    def place(
        self,
        vn: Data,
        sn_state: Data,  # noqa: ARG002
        env: SimuVNEEnv,
        *,
        context: StrategyContext,
    ) -> StrategyResult:
        temp_env = copy.deepcopy(env)
        allocator = GAL2Allocator(temp_env)
        success, mapping, _ = allocator.greedy_place(vn)
        metadata: Dict[str, object] = {"placed_with": "GAL-PN"}
        if not success:
            metadata["failure_reason"] = "GAL-PN greedy placement failed"
        if context.verbose:
            msg = "✓" if success else "✗"
            print(f"[GAL-PN] {msg} step={context.step_id}, vn_nodes={vn.x.size(0)}")
        return StrategyResult(success=success, mapping=mapping if success else {}, metadata=metadata)


def run_gal2_strategy_test(
    *,
    detail_print: bool = False,
    tester_cls: Type[SingleTester] = SingleTester,
    config: Optional[TestConfig] = None,
    config_overrides: Optional[Dict[str, object]] = None,
    printer: Optional[TestPrinter] = None,
):
    """兼容 run_single_strategy_test 的 GAL_2 封装。"""

    from tests.test_strategy import run_single_strategy_test

    return run_single_strategy_test(
        strategy_factory=GAL2PlacementStrategy,
        tester_cls=tester_cls,
        detail_print=detail_print,
        config=config,
        config_overrides=config_overrides,
        printer=printer,
        strategy_label="gal_pn",
    )


def smoke_test_gal2_strategy(*, detail_print: bool = False) -> Dict[str, any]:
    """GAL_2 的小参数冒烟测试。"""

    config = get_smoke_config()

    printer = TestPrinter(
        enable_logging=True,
        enable_plotting=False,
        test_scope="gal_pn_single",
    )
    try:
        printer.start_round(
            table_title="GAL-2 Strategy Smoke Test",
            config_info=format_config_info(
                config,
                workflow_keys=config.workflow_types.keys(),
            ),
        )
        result = run_strategy_with_details(
            strategy_factory=GAL2PlacementStrategy,
            config=config,
            printer=printer,
            detail_print=detail_print,
        )
        printer.add_row(
            build_strategy_row("gal_pn", result),
            strategy_info={"strategy_name": "GAL2PlacementStrategy"},
        )
        printer.finalize()
    finally:
        printer.close()

    return result


if __name__ == "__main__":
    summary = smoke_test_gal2_strategy(detail_print=False)["summary"]
    print(
        "[GAL2Smoke] total_tasks={total}, accepted={accepted}, acceptance_rate={rate:.2f}%".format(
            total=summary["total_tasks"],
            accepted=summary["accepted"],
            rate=summary["acceptance_rate"] * 100.0,
        )
    )


