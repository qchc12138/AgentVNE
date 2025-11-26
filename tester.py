from __future__ import annotations
import warnings
warnings.filterwarnings(
    "ignore",
    message="An issue occurred while importing 'torch-scatter'. Disabling its usage.",
)
warnings.filterwarnings(
    "ignore",
    message="An issue occurred while importing 'torch-cluster'. Disabling its usage.",
)
warnings.filterwarnings(
    "ignore",
    message="An issue occurred while importing 'torch-sparse'. Disabling its usage.",
)
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
)

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from tests.test_printer import TestPrinter
from tests.test_strategy import TestConfig, format_config_info, run_single_strategy_test
from tests.test_configs import (
    ParameterSpec,
    DEFAULT_SN_TOPOLOGY,
    DEFAULT_WORKFLOW_TYPES,
    parse_workflows,
    parse_parameters,
    build_round_configs,
)
from tests.tester_ga import GAPlacementStrategy
from tests.tester_gal import GALPlacementStrategy
from tests.tester_gal_2 import GAL2PlacementStrategy
from tests.tester_gal_3 import GAL3PlacementStrategy
from tests.tester_null import NullPlacementStrategy
from tests.tester_ft1 import get_finetuned_tester_registry
from tests.tester_ft1 import FT1PlacementStrategy


StrategyFactory = Callable[[], "PlacementStrategy"]

# 直接引用协议而不产生循环导入
from tests.test_strategy import PlacementStrategy  # noqa: E402


def resolve_strategies(names: Optional[Iterable[str]]) -> Dict[str, StrategyFactory]:
    base_registry: Dict[str, StrategyFactory] = {
        "null": lambda: NullPlacementStrategy(),
        "ga": lambda: GAPlacementStrategy(),
        "gal": lambda: GALPlacementStrategy(),
        "gal2": lambda: GAL2PlacementStrategy(),
        "gal3": lambda: GAL3PlacementStrategy(),
        "ft1": lambda: FT1PlacementStrategy(),
    }
    finetuned_registry = get_finetuned_tester_registry()
    full_registry = dict(base_registry)
    full_registry.update(finetuned_registry)

    if not names:
        return dict(base_registry)

    resolved: Dict[str, StrategyFactory] = {}
    for raw_name in names:
        key = raw_name.strip().lower()
        if key not in full_registry:
            raise ValueError(
                f"未知策略 '{raw_name}'，当前支持: {', '.join(full_registry)}"
            )
        resolved[key] = full_registry[key]
    return resolved


def build_config(
    *,
    sn_topology: str,
    workflows: Dict[str, str],
    base_device: str,
    base_penalty: float,
    params: ParameterSpec,
) -> TestConfig:
    return TestConfig(
        sn_topology_path=sn_topology,
        workflow_types=workflows,
        arrival_rate=params.arrival_rate,
        mean_lifetime=params.mean_lifetime,
        max_time_steps=params.max_time_steps,
        device=params.device or base_device,
        seed=params.seed,
        penalty=params.penalty if params.penalty is not None else base_penalty,
    )


def build_round_configs(
    *,
    parameter_specs: List[ParameterSpec],
    sn_topology: str,
    workflows: Dict[str, str],
    base_device: str,
    base_penalty: float,
) -> List[TestConfig]:
    if not parameter_specs:
        raise ValueError("至少需要一组测试参数。")

    return [
        build_config(
            sn_topology=sn_topology,
            workflows=workflows,
            base_device=base_device,
            base_penalty=base_penalty,
            params=spec,
        )
        for spec in parameter_specs
    ]


class Tester:
    """负责执行多参数 + 多策略的对比实验。"""

    def __init__(
        self,
        *,
        strategy_factories: Dict[str, StrategyFactory],
        round_configs: List[TestConfig],
        detail_print: bool = False,
        enable_logging: bool = True,
        enable_plotting: bool = False,
        output_dir: Optional[str] = None,
        session_name: Optional[str] = None,
    ) -> None:
        self.strategy_factories = strategy_factories
        self.round_configs = round_configs
        self.detail_print = detail_print
        self.enable_logging = enable_logging
        self.enable_plotting = enable_plotting
        self.output_dir = output_dir
        self.session_name = session_name

    def run(self) -> None:
        printer = TestPrinter(
            enable_logging=self.enable_logging,
            enable_plotting=self.enable_plotting,
            output_dir=self.output_dir,
            session_name=self.session_name,
            test_scope="tester",
        )
        try:
            for idx, config in enumerate(self.round_configs, start=1):
                table_title = f"Tester 参数组 #{idx}"
                printer.start_round(
                    table_title=table_title,
                    config_info=format_config_info(
                        config,
                        workflow_keys=config.workflow_types.keys(),
                    ),
                )
                for label, factory in self.strategy_factories.items():
                    run_single_strategy_test(
                        strategy_factory=factory,
                        config=config,
                        detail_print=self.detail_print,
                        printer=printer,
                        strategy_label=label,
                    )
                printer.finalize()
        finally:
            printer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tester: 多策略、多参数测试控制器")
    parser.add_argument(
        "--sn-topology",
        default=DEFAULT_SN_TOPOLOGY,
        help="SN 拓扑文件路径",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        help="workflow_name=path，可重复指定",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        help="策略名称，可重复指定；默认加载全部",
    )
    parser.add_argument(
        "--parameter",
        action="append",
        dest="parameters",
        help="参数组，如 arrival_rate=0.5,mean_lifetime=50,max_time_steps=100,seed=2025，可重复指定",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.2,
        help="默认 arrival_rate（当未提供 --parameter 时生效）",
    )
    parser.add_argument(
        "--mean-lifetime",
        type=float,
        default=20.0,
        help="默认 mean_lifetime（当未提供 --parameter 时生效）",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=100,
        help="默认 max_time_steps（当未提供 --parameter 时生效）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="默认随机种子（当未提供 --parameter 时生效）",
    )
    parser.add_argument("--device", default="cpu", help="默认运行设备")
    parser.add_argument(
        "--penalty",
        type=float,
        default=-150.0,
        help="映射失败惩罚",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="是否打印单策略运行的详细信息",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="关闭 TestPrinter 日志输出",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="启用图表生成",
    )
    parser.add_argument("--output-dir", help="自定义输出目录")
    parser.add_argument("--session-name", help="自定义会话名称")
    return parser


def create_tester_from_args(args: argparse.Namespace) -> Tester:
    workflows = parse_workflows(args.workflow)
    if args.parameters:
        parameter_specs = parse_parameters(args.parameters)
    else:
        parameter_specs = [
            ParameterSpec(
                arrival_rate=args.arrival_rate,
                mean_lifetime=args.mean_lifetime,
                max_time_steps=args.max_time_steps,
                seed=args.seed,
                device=args.device,
                penalty=args.penalty,
            )
        ]
    strategies = resolve_strategies(args.strategy)
    round_configs = build_round_configs(
        parameter_specs=parameter_specs,
        sn_topology=args.sn_topology,
        workflows=workflows,
        base_device=args.device,
        base_penalty=args.penalty,
    )
    tester = Tester(
        strategy_factories=strategies,
        round_configs=round_configs,
        detail_print=args.detail,
        enable_logging=not args.disable_logging,
        enable_plotting=args.plot,
        output_dir=args.output_dir,
        session_name=args.session_name,
    )
    return tester


def main(argv: Optional[List[str]] = None) -> None:
    """
    默认入口：main 内编码最简策略/参数，便于直接修改后运行。
    如需 CLI 控制，可传入 argv（命令行执行时自动覆盖）。
    """

    manual_strategies = [
        "null",
        "ga",
        "gal",
        "gal2",
        "gal3",
        "ft1",
    ]
    manual_parameters = [
        {
            "arrival_rate": 0.2,
            "mean_lifetime": 20,
            "max_time_steps": 100,
            "seed": 2025,
        },
        {
            "arrival_rate": 0.5,
            "mean_lifetime": 50,
            "max_time_steps": 1000,
            "seed": 2025,
        },
        {
            "arrival_rate": 1.0,
            "mean_lifetime": 100,
            "max_time_steps": 1000,
            "seed": 2025,
        },
    ]

    parser = build_arg_parser()
    if argv is None:
        default_cli: List[str] = []
        for strategy_name in manual_strategies:
            default_cli.extend(["--strategy", strategy_name])
        for param_dict in manual_parameters:
            param_str = ",".join(f"{k}={v}" for k, v in param_dict.items())
            default_cli.extend(["--parameter", param_str])
        args = parser.parse_args(default_cli)
    else:
        args = parser.parse_args(argv)

    tester = create_tester_from_args(args)
    tester.run()


if __name__ == "__main__":
    main()

