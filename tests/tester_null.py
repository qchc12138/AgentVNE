from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import time
import os
import sys
from pathlib import Path

import numpy as np
from torch_geometric.data import Data

#region sys.path 管理
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env import SimuVNEEnv, WorkflowGenerator
from tests.test_printer import TestPrinter
from tests.test_configs import get_smoke_config
from tests.test_strategy import (
    PlacementStrategy,
    SingleTester,
    StrategyContext,
    StrategyResult,
    TestConfig,
    format_config_info,
    run_single_strategy_test,
    run_strategy_with_details,
    build_strategy_row,
)

__all__ = [
    "NullPlacementStrategy",
    "run_null_strategy_test",
    "smoke_test_null_strategy",
]

# 获取项目根目录（tester_null.py 在 tests/ 目录下，_PROJECT_ROOT 已在文件开头定义）
DEFAULT_SN_TOPOLOGY = str(_PROJECT_ROOT / 'topo' / 'SN_topology_2.json')
DEFAULT_WORKFLOWS = {
    "workflow1": str(_PROJECT_ROOT / 'workflow_topo' / 'workflow1_topo.json'),
}


#region NullPlacementStrategy
@dataclass
class NullPlacementStrategy(PlacementStrategy):
    """空策略：拒绝所有任务，不对环境做任何修改，用于验证基架。"""

    name: str = "null"

    def prepare(self, env: SimuVNEEnv) -> None:  # noqa: D401, ARG002
        """空策略无需任何准备。"""
        return None

    def place(
        self,
        vn: Data,
        sn_state: Data,  # noqa: ARG002
        env: SimuVNEEnv,  # noqa: ARG002
        *,
        context: StrategyContext,
    ) -> StrategyResult:
        if context.verbose:
            print(
                f"[NullStrategy] 拒绝任务: VN节点数={vn.x.size(0)}, step={context.step_id}"
            )
        metadata: Dict[str, object] = {
            "reason": "NullStrategy 总是拒绝任务",
            "vn_nodes": vn.x.size(0),
        }
        return StrategyResult(success=False, mapping={}, metadata=metadata)


#endregion NullPlacementStrategy


#region 运行封装
def run_null_strategy_test(
    *,
    detail_print: bool = False,
    tester_cls: Type[SingleTester] = SingleTester,
    config: Optional[TestConfig] = None,
    config_overrides: Optional[Dict[str, object]] = None,
    printer: Optional[TestPrinter] = None,
    strategy_label: str = "null",
):
    """统一封装 Null 策略的单次测试。"""

    return run_single_strategy_test(
        strategy_factory=NullPlacementStrategy,
        tester_cls=tester_cls,
        detail_print=detail_print,
        config=config,
        config_overrides=config_overrides,
        printer=printer,
        strategy_label=strategy_label,
    )


def smoke_test_null_strategy(*, detail_print: bool = True):
    """较小参数的冒烟测试，使用 TestPrinter 记录日志。"""

    config = get_smoke_config()

    printer = TestPrinter(
        enable_logging=True,
        enable_plotting=False,
        test_scope="null_single",
    )
    try:
        printer.start_round(
            table_title="Null Strategy Smoke Test",
            config_info=format_config_info(
                config,
                workflow_keys=config.workflow_types.keys(),
            ),
        )
        result = run_strategy_with_details(
            strategy_factory=NullPlacementStrategy,
            config=config,
            printer=printer,
            detail_print=detail_print,
    )
        printer.add_row(
            build_strategy_row("null", result),
            strategy_info={"strategy_name": "NullPlacementStrategy"},
        )
        printer.finalize()
    finally:
        printer.close()

    return result


#endregion 运行封装


if __name__ == "__main__":
    summary = smoke_test_null_strategy(detail_print=True)["summary"]
    print(
        "[NullSmoke] total_tasks={total}, accepted={accepted}, acceptance_rate={rate:.2f}%".format(
            total=summary["total_tasks"],
            accepted=summary["accepted"],
            rate=summary["acceptance_rate"] * 100.0,
        )
    )

