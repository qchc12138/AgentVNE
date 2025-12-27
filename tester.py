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
import csv
import os
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
from tests.tester_greedy import GreedyPlacementStrategy
from tests.tester_gal import GALPlacementStrategy
from tests.tester_null import NullPlacementStrategy
from tests.tester_ft1 import get_finetuned_tester_registry
from tests.tester_ft1 import FT1PlacementStrategy
from tests.tester_ft_n import FTNPlacementStrategy


StrategyFactory = Callable[[], "PlacementStrategy"]

# 直接引用协议而不产生循环导入
from tests.test_strategy import PlacementStrategy  # noqa: E402


def resolve_strategies(names: Optional[Iterable[str]]) -> Dict[str, StrategyFactory]:
    base_registry: Dict[str, StrategyFactory] = {
        "null": lambda: NullPlacementStrategy(),
        "ga": lambda: GAPlacementStrategy(),
        "gal-vne": lambda: GALPlacementStrategy(),  # 原 gal，改名为 GAL-VNE（小写）
        "gal": lambda: GALPlacementStrategy(),  # 向后兼容别名
        # "gal_pn": lambda: GALPNPlacementStrategy(),  # 已注释
        "greedy": lambda: GALSNPlacementStrategy(),  # 原 gal_sn，改名为 greedy
        "gal_sn": lambda: GALSNPlacementStrategy(),  # 向后兼容别名
        "ft1": lambda: FT1PlacementStrategy(),  # FineTuned
        "finetuned": lambda: FT1PlacementStrategy(),  # FineTuned 别名
        "ft_n": lambda: FTNPlacementStrategy(),  # Pretrain
        "pretrain": lambda: FTNPlacementStrategy(),  # Pretrain 别名
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
        # 记录每个轮次的任务接收情况：round_task_acceptance[round_idx][task_id][strategy] = 1 or 0
        self.round_task_acceptance: Dict[int, Dict[int, Dict[str, int]]] = {}
        # 保存 printer 对象，用于获取 session_dir
        self.printer: Optional["TestPrinter"] = None
    
    def _save_task_acceptance_csv(self, round_idx: int) -> None:
        """保存任务接收情况到CSV文件"""
        if round_idx not in self.round_task_acceptance:
            return
        
        round_data = self.round_task_acceptance[round_idx]
        if not round_data:
            return
        
        # 获取所有策略名称（按策略工厂的顺序）
        strategy_names = list(self.strategy_factories.keys())
        
        # 获取所有任务ID并排序
        task_ids = sorted(round_data.keys())
        
        # 确定输出文件路径：使用 printer 的 session_dir
        if self.printer and self.printer.session_dir:
            csv_dir = self.printer.session_dir
        elif self.output_dir:
            csv_dir = self.output_dir
        else:
            csv_dir = "."
        
        # 构建文件名
        csv_filename = f"round_{round_idx}_task_acceptance.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        # 确保目录存在
        os.makedirs(csv_dir, exist_ok=True)
        
        # 写入CSV文件
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(strategy_names)
            # 写入每行数据（每个任务）
            for task_id in task_ids:
                row = []
                for strategy_name in strategy_names:
                    # 获取该策略对该任务的接收情况，如果不存在则记为0
                    acceptance = round_data[task_id].get(strategy_name, 0)
                    row.append(acceptance)
                writer.writerow(row)
        
        print(f"  ✓ 任务接收情况已保存到: {csv_path}")

    def run(self) -> None:
        printer = TestPrinter(
            enable_logging=self.enable_logging,
            enable_plotting=self.enable_plotting,
            output_dir=self.output_dir,
            session_name=self.session_name,
            test_scope="tester",
        )
        self.printer = printer  # 保存 printer 对象，用于获取 session_dir
        try:
            total_rounds = len(self.round_configs)
            total_strategies = len(self.strategy_factories)
            print(f"\n{'='*80}")
            print(f"开始测试：共 {total_rounds} 轮，每轮 {total_strategies} 个策略")
            print(f"{'='*80}\n")
            
            for idx, config in enumerate(self.round_configs, start=1):
                print(f"\n[轮次 {idx}/{total_rounds}] 开始测试")
                print(f"  参数配置: arrival_rate={config.arrival_rate}, "
                      f"mean_lifetime={config.mean_lifetime}, "
                      f"max_time_steps={config.max_time_steps}, seed={config.seed}")
                
                table_title = f"Tester Param Group #{idx}"
                printer.start_round(
                    table_title=table_title,
                    config_info=format_config_info(
                        config,
                        workflow_keys=config.workflow_types.keys(),
                    ),
                )
                
                # 初始化当前轮次的任务接收记录
                self.round_task_acceptance[idx] = {}
                
                strategy_list = list(self.strategy_factories.items())
                for strategy_idx, (label, factory) in enumerate(strategy_list, start=1):
                    print(f"  [{strategy_idx}/{total_strategies}] 正在测试策略: {label}")
                    result = run_single_strategy_test(
                        strategy_factory=factory,
                        config=config,
                        detail_print=self.detail_print,
                        printer=printer,
                        strategy_label=label,
                    )
                    # 收集任务接收情况
                    tasks = result.get("tasks", [])
                    for task in tasks:
                        task_id = task.get("task_id")
                        if task_id is not None:
                            if task_id not in self.round_task_acceptance[idx]:
                                self.round_task_acceptance[idx][task_id] = {}
                            # 被接受记为1，否则记为0
                            self.round_task_acceptance[idx][task_id][label] = 1 if task.get("success", False) else 0
                    print(f"  [{strategy_idx}/{total_strategies}] ✓ 策略 {label} 测试完成")
                
                print(f"\n[轮次 {idx}/{total_rounds}] 所有策略测试完成，生成汇总结果...")
                printer.finalize()
                # 保存任务接收情况到CSV文件
                self._save_task_acceptance_csv(idx)
                print(f"[轮次 {idx}/{total_rounds}] ✓ 轮次完成\n")
            
            print(f"\n{'='*80}")
            print(f"所有测试完成：共完成 {total_rounds} 轮测试")
            print(f"{'='*80}\n")
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
    
    可以直接运行：python3 tester.py
    默认会测试所有策略并生成对比图表。
    
    如需 CLI 控制，可传入 argv（命令行执行时自动覆盖）。
    """

    # 默认测试所有策略（不包括 null，因为它是测试用的占位策略）
    manual_strategies = [
        "ga",           # 遗传算法
        "gal-vne",      # 贪心分配算法（基于 noderank），原 gal（注意：使用小写）
        # "gal_pn",     # 贪心分配算法（逐节点），已注释
        "greedy",       # 贪心分配算法（SN 排序），原 gal_sn
        "pretrain",     # 预训练模型（别名：ft_n）
        "finetuned",    # 微调模型（别名：ft1）
    ]
    
    # 默认测试参数配置
    # 可添加多个字典，用于多轮测试（每轮所有策略使用相同参数）
    manual_parameters = [
        # {
        #    "arrival_rate": 0.05,
        #    "mean_lifetime": 25,
        #    "max_time_steps": 1000,
        #    "seed": 42,  # 不同轮次可以使用不同 seed
        # },

        # {
        #    "arrival_rate": 0.25,
        #    "mean_lifetime": 20,
        #    "max_time_steps": 1000,
        #    "seed": 42,  # 不同轮次可以使用不同 seed
        # },
        # # {
        # #    "arrival_rate": 0.2,
        # #    "mean_lifetime": 25,
        # #    "max_time_steps": 1000,
        # #    "seed": 42,  # 不同轮次可以使用不同 seed
        # # },
        {
           "arrival_rate": 0.25,
           "mean_lifetime": 40,
           "max_time_steps": 11000,
           "seed": 42,  # 不同轮次可以使用不同 seed
        },
        # {
        #    "arrival_rate": 0.4,
        #    "mean_lifetime": 25,
        #    "max_time_steps": 1000,
        #    "seed": 42,  # 不同轮次可以使用不同 seed
        # },
        # {
        #    "arrival_rate": 0.25,
        #    "mean_lifetime": 40,
        #    "max_time_steps": 1000,
        #    "seed": 42,  # 不同轮次可以使用不同 seed
        # }
        # {
        #     "arrival_rate": 0.2,      # 任务到达率（泊松分布）
        #     "mean_lifetime": 15,      # 平均生存时间（指数分布）
        #     "max_time_steps": 1000,   # 最大时间步数
        #     "seed": 42,             # 随机种子（保证不同策略使用相同的任务序列）
        # },
        # # 可以取消注释以下配置来添加更多测试轮次
        # {
        #    "arrival_rate": 0.2,
        #    "mean_lifetime": 20,
        #    "max_time_steps": 1000,
        #    "seed": 42,  # 不同轮次可以使用不同 seed
        # },
        # {
        #    "arrival_rate": 0.2,
        #    "mean_lifetime": 25,
        #    "max_time_steps": 1000,
        #    "seed": 42,  # 不同轮次可以使用不同 seed
        # },
        # {
        #    "arrival_rate": 0.2,
        #    "mean_lifetime": 30,
        #    "max_time_steps": 1000,
        #    "seed": 42,  # 不同轮次可以使用不同 seed
        # },
        # {
        #    "arrival_rate": 0.2,
        #    "mean_lifetime": 35,
        #    "max_time_steps": 1000,
        #    "seed": 42,
        # },
        # {
        #    "arrival_rate": 0.2,
        #    "mean_lifetime": 40,
        #    "max_time_steps": 1000,
        #    "seed": 42,
        # }
    ]

    parser = build_arg_parser()
    if argv is None:
        # 构建默认命令行参数
        default_cli: List[str] = []
        # 默认开启绘图功能，生成对比图和汇总趋势图
        default_cli.append("--plot")
        # 添加所有策略
        for strategy_name in manual_strategies:
            default_cli.extend(["--strategy", strategy_name])
        # 添加所有参数配置
        for param_dict in manual_parameters:
            param_str = ",".join(f"{k}={v}" for k, v in param_dict.items())
            default_cli.extend(["--parameter", param_str])
        args = parser.parse_args(default_cli)
    else:
        # 使用命令行参数（覆盖默认配置）
        args = parser.parse_args(argv)

    tester = create_tester_from_args(args)
    tester.run()


if __name__ == "__main__":
    main()

