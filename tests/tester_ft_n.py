from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from torch_geometric.data import Data
import sys

#region sys.path 管理
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
#endregion

from env import SimuVNEEnv
from fine_tuning_1 import PPOAgent, ValueNet
from model_1 import SimuVNE as SimuVNEModel1
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

# 复用 tester_ft1.py 中的方法
from tests.tester_ft1 import (
    GreedyPPOAgent,
    extract_model_name,
)

__all__ = [
    "PretrainedPlacementStrategy",
    "FTNPlacementStrategy",
    "find_latest_pretrain_checkpoint",
    "resolve_pretrain_checkpoint",
    "load_pretrain_policy",
    "run_pretrained_strategy_test",
    "smoke_test_pretrained_strategy",
    "pretrained_strategy_factory",
]

#region 常量定义
DEFAULT_PRETRAIN_OUTPUT_DIR = "/home/zrz/AgentVNE/AgentVNE/pretrain_outputs"
PRETRAINED_MODEL_NAME = "model_1"
MODEL_FACTORIES = {
    PRETRAINED_MODEL_NAME: SimuVNEModel1,
}
DEFAULT_CHECKPOINT_FILENAME = "checkpoint_latest.pt"
#endregion


#region 辅助函数
def find_latest_pretrain_checkpoint(output_dir: str = DEFAULT_PRETRAIN_OUTPUT_DIR) -> Optional[str]:
    """
    查找预训练模型的最新检查点文件。
    
    优先查找 checkpoint_latest.pt，如果不存在则查找 checkpoint_best.pt。
    
    参数:
        output_dir: 预训练输出根目录，默认为 DEFAULT_PRETRAIN_OUTPUT_DIR
    
    返回:
        检查点文件的完整路径，如果未找到则返回 None
    """
    if not os.path.isdir(output_dir):
        return None
    
    # 查找 checkpoint_latest.pt
    latest_checkpoint = os.path.join(output_dir, "checkpoint_latest.pt")
    if os.path.isfile(latest_checkpoint):
        return latest_checkpoint
    # 查找 checkpoint_best.pt
    best_checkpoint = os.path.join(output_dir, "checkpoint_best.pt")
    if os.path.isfile(best_checkpoint):
        return best_checkpoint
    
    
    return None


def resolve_pretrain_checkpoint(
    path: Optional[str], *, filename: str = DEFAULT_CHECKPOINT_FILENAME, output_dir: str = DEFAULT_PRETRAIN_OUTPUT_DIR
) -> Optional[str]:
    """
    解析预训练模型检查点路径。
    
    如果 path 为 None，自动查找最新检查点文件。
    如果 path 是目录，自动拼接 filename。
    如果 path 是文件，直接返回。
    
    参数:
        path: 模型文件路径、目录路径或 None
        filename: 当 path 是目录或 None 时，要查找的文件名（默认 "checkpoint_best.pt"）
        output_dir: 预训练输出根目录（仅在 path=None 时使用）
    
    返回:
        规范化后的完整文件路径，如果未找到则返回 None
    """
    if path:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            candidate = os.path.join(path, filename)
            return candidate if os.path.isfile(candidate) else None
        return path if os.path.isfile(path) else None
    
    # 自动查找最新检查点
    return find_latest_pretrain_checkpoint(output_dir)


def load_pretrain_policy(path: str, device: str) -> tuple[Any, Dict[str, object]]:
    """
    加载预训练策略网络模型。
    
    参数:
        path: 预训练检查点文件路径
        device: 目标设备（"cpu" 或 "cuda"）
    
    返回:
        (policy, policy_cfg): 加载的策略网络实例和模型配置字典
    
    异常:
        FileNotFoundError: 如果检查点文件不存在
        RuntimeError: 如果加载失败（如 state_dict 不匹配）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到预训练模型文件: {path}")
    
    factory = MODEL_FACTORIES[PRETRAINED_MODEL_NAME]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    
    # 如果配置为空，使用默认值
    if not model_cfg:
        model_cfg = {
            "input_dim": 6,
            "hidden_dim": 64,
            "hist_dim": 32,
            "num_nodes_j": 10,
        }
    
    model = factory(
        input_dim=model_cfg.get("input_dim", 6),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        hist_dim=model_cfg.get("hist_dim", 32),
        num_nodes_j=model_cfg.get("num_nodes_j", 10),
    )
    model.load_state_dict(state_dict, strict=False)  # 预训练模型可能结构不完全匹配，使用 strict=False
    model.to(device)
    model.eval()
    return model, model_cfg
#endregion


#region PretrainedPlacementStrategy
@dataclass
class PretrainedPlacementStrategy(PlacementStrategy):
    """
    基于预训练模型的放置策略。
    
    预训练模型通常只包含策略网络，没有价值网络。因此会创建一个随机初始化的价值网络
    用于 PPOAgent（虽然价值网络在测试时不会被使用）。
    
    仅在内部副本环境上执行搜索与资源扣减，主环境资源更新由测试框架负责。
    """

    name: str = "pretrained"
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    k_hop: int = 1
    verbose: bool = False
    pretrain_output_dir: str = DEFAULT_PRETRAIN_OUTPUT_DIR

    def __post_init__(self):
        """初始化时解析模型文件路径。"""
        self._checkpoint_file = resolve_pretrain_checkpoint(
            self.checkpoint_path,
            filename=DEFAULT_CHECKPOINT_FILENAME,
            output_dir=self.pretrain_output_dir,
        )
        self.agent: Optional[PPOAgent] = None
        # 提取模型名称用于日志记录
        self.model_name_display = extract_model_name(self._checkpoint_file)

    def prepare(self, env: SimuVNEEnv) -> None:
        """
        加载预训练策略网络，创建随机初始化的价值网络，初始化 PPOAgent。
        
        异常:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
        """
        if not self._checkpoint_file:
            raise FileNotFoundError(
                "未找到预训练模型，请先运行 pretrain_1.py 进行预训练或指定 --checkpoint-path。"
                f"检查点文件: {self._checkpoint_file}"
            )
        
        # 打印模型加载信息
        print(f"[Pretrained] 加载模型: {self.model_name_display} ({PRETRAINED_MODEL_NAME})")
        print(f"  检查点: {os.path.basename(self._checkpoint_file)}")
        
        try:
            policy, cfg = load_pretrain_policy(self._checkpoint_file, self.device)
            # 预训练模型没有价值网络，创建一个随机初始化的价值网络
            # （虽然测试时不会使用价值网络，但 PPOAgent 需要它）
            value_net = ValueNet(
                input_dim=int(cfg.get("input_dim", 6)),
                hidden_dim=int(cfg.get("hidden_dim", 64)),
            )
            value_net.to(self.device)
            value_net.eval()
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(
                f"模型加载失败: {e}。"
                f"请确保模型文件存在且格式正确。"
            ) from e
        
        self.agent = GreedyPPOAgent(policy, value_net, device=self.device)
        self.agent.policy.eval()
        self.agent.value_net.eval()

    def place(
        self,
        vn: Data,
        sn_state: Data,
        env: SimuVNEEnv,
        *,
        context: StrategyContext,
    ) -> StrategyResult:
        """
        执行放置决策：调用 PPOAgent.act 方法生成映射。
        
        返回:
            StrategyResult: 包含成功标志、映射字典和元数据（logprob、value 等）
        
        异常:
            RuntimeError: 如果策略尚未 prepare
        """
        if self.agent is None:
            raise RuntimeError(
                "PretrainedPlacementStrategy 尚未准备，请先调用 prepare()。"
            )
        
        sim_env = copy.deepcopy(env)
        mapping, logprob, value = self.agent.act(
            vn,
            sn_state,
            env=sim_env,
            k_hop=self.k_hop,
            verbose=self.verbose or context.verbose,
        )
        
        success = len(mapping) == vn.x.size(0)
        metadata: Dict[str, Any] = {
            "logprob": float(logprob),
            "value": float(value),
            "strategy": self.name,
            "model_name": self.model_name_display,
        }
        
        # 保存调试信息（如果可用）
        debug_info = getattr(self.agent, "_last_debug_info", None)
        if debug_info:
            if success:
                metadata["success_debug"] = {
                    "probs_matrix": debug_info.get("probs_matrix"),
                    "final_mapping": debug_info.get("final_mapping", {}),
                    "sn_snapshot": debug_info.get("sn_snapshot"),
                }
            else:
                metadata["failure_debug"] = debug_info
        
        return StrategyResult(success, mapping if success else {}, metadata)
#endregion


class FTNPlacementStrategy(PretrainedPlacementStrategy):
    """
    无需显式参数的预设策略，自动从默认目录选择最新预训练模型。
    """

    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        k_hop: int = 1,
        verbose: bool = False,
        pretrain_output_dir: str = DEFAULT_PRETRAIN_OUTPUT_DIR,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            k_hop=k_hop,
            verbose=verbose,
            pretrain_output_dir=pretrain_output_dir,
        )


#region 工厂函数
def pretrained_strategy_factory(
    *,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    k_hop: int = 1,
    verbose: bool = False,
    pretrain_output_dir: str = DEFAULT_PRETRAIN_OUTPUT_DIR,
) -> PlacementStrategy:
    """
    创建 PretrainedPlacementStrategy 实例的工厂函数。
    
    默认情况下，如果 checkpoint_path 为 None，会自动查找最新预训练检查点。
    
    参数:
        checkpoint_path: 预训练检查点路径（None 时自动查找最新检查点）
        device: 运行设备
        k_hop: BFS k 跳参数
        verbose: 是否详细输出
        pretrain_output_dir: 预训练输出根目录
    
    返回:
        PretrainedPlacementStrategy 实例
    """
    return PretrainedPlacementStrategy(
        checkpoint_path=checkpoint_path,
        device=device,
        k_hop=k_hop,
        verbose=verbose,
        pretrain_output_dir=pretrain_output_dir,
    )
#endregion


#region 运行封装
def run_pretrained_strategy_test(
    *,
    detail_print: bool = False,
    tester_cls: Type[SingleTester] = SingleTester,
    config: Optional[TestConfig] = None,
    config_overrides: Optional[Dict[str, object]] = None,
    printer: Optional[TestPrinter] = None,
    strategy_label: str = "pretrained",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    k_hop: int = 1,
    pretrain_output_dir: str = DEFAULT_PRETRAIN_OUTPUT_DIR,
):
    """
    统一封装 Pretrained 策略的单次测试。
    
    参数:
        detail_print: 是否打印详细过程
        tester_cls: Tester 类（默认 SingleTester）
        config: 测试配置（可选）
        config_overrides: 配置覆盖字典（可选）
        printer: TestPrinter 实例（可选）
        strategy_label: 策略标签（默认 "pretrained"）
        checkpoint_path: 预训练检查点路径（None 时自动查找最新检查点）
        device: 运行设备
        k_hop: BFS k 跳参数
        pretrain_output_dir: 预训练输出根目录
    
    返回:
        测试结果字典（包含 tasks、summary、lag_time 等）
    
    异常:
        FileNotFoundError: 如果未找到模型文件
    """
    factory = lambda: pretrained_strategy_factory(
        checkpoint_path=checkpoint_path,
        device=device,
        k_hop=k_hop,
        verbose=False,
        pretrain_output_dir=pretrain_output_dir,
    )
    
    return run_single_strategy_test(
        strategy_factory=factory,
        tester_cls=tester_cls,
        detail_print=detail_print,
        config=config,
        config_overrides=config_overrides,
        printer=printer,
        strategy_label=strategy_label,
    )


def smoke_test_pretrained_strategy(
    *,
    detail_print: bool = False,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    k_hop: int = 1,
    pretrain_output_dir: str = DEFAULT_PRETRAIN_OUTPUT_DIR,
) -> Optional[Dict[str, Any]]:
    """
    较小参数的 Pretrained 冒烟测试，输出详细日志。
    
    如果找不到模型文件，将打印错误信息并返回 None。
    
    参数:
        detail_print: 是否打印详细过程
        checkpoint_path: 预训练检查点路径（None 时自动查找最新检查点）
        device: 运行设备
        k_hop: BFS k 跳参数
        pretrain_output_dir: 预训练输出根目录
    
    返回:
        测试结果字典，如果未找到模型则返回 None
    """
    # 尝试创建策略实例以检查模型文件是否存在
    strategy = PretrainedPlacementStrategy(
        checkpoint_path=checkpoint_path,
        device=device,
        k_hop=k_hop,
        verbose=detail_print,
        pretrain_output_dir=pretrain_output_dir,
    )
    
    if not strategy._checkpoint_file:
        print("[Pretrained] ❌ 未找到模型文件，已跳过测试。")
        print(f"[Pretrained] 检查点文件: {strategy._checkpoint_file}")
        print(f"[Pretrained] 请确保已运行 pretrain_1.py 进行预训练，或手动指定 --checkpoint-path。")
        return None

    config = get_smoke_config()

    printer = TestPrinter(
        enable_logging=True,
        enable_plotting=False,
        test_scope="pretrained_single",
    )
    try:
        printer.start_round(
            table_title="Pretrained Strategy Smoke Test",
            config_info=format_config_info(
                config,
                workflow_keys=config.workflow_types.keys(),
            ),
        )
        result = run_strategy_with_details(
            strategy_factory=lambda: strategy,
            config=config,
            printer=printer,
            detail_print=detail_print,
        )
        printer.add_row(
            build_strategy_row(strategy.name, result),
            strategy_info={
                "strategy_name": "PretrainedPlacementStrategy",
                "model_name": strategy.model_name_display,
            },
        )
        printer.finalize()
    finally:
        printer.close()

    return result
#endregion


#region CLI 接口
def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。
    
    返回:
        argparse.ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(description="Pretrained 策略冒烟测试")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="预训练检查点路径（checkpoint_best.pt 或其所在目录，默认：自动查找最新检查点）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="运行设备（默认：cpu）",
    )
    parser.add_argument(
        "--k-hop",
        type=int,
        default=1,
        help="BFS k 跳参数（默认：1）",
    )
    parser.add_argument(
        "--pretrain-output-dir",
        type=str,
        default=DEFAULT_PRETRAIN_OUTPUT_DIR,
        help="预训练输出根目录",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="是否打印详细日志",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """
    主函数：运行 Pretrained 策略的冒烟测试。
    
    参数:
        argv: 命令行参数列表（可选，用于测试）
    """
    args = build_arg_parser().parse_args(argv)
    result = smoke_test_pretrained_strategy(
        detail_print=args.detail,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        k_hop=args.k_hop,
        pretrain_output_dir=args.pretrain_output_dir,
    )
    
    if result is not None:
        summary = result["summary"]
        print(
            "[PretrainedSmoke] total_tasks={total}, accepted={accepted}, acceptance_rate={rate:.2f}%".format(
                total=summary["total_tasks"],
                accepted=summary["accepted"],
                rate=summary["acceptance_rate"] * 100.0,
            )
        )
    else:
        print("[PretrainedSmoke] 测试已取消：未找到模型文件。")
#endregion


if __name__ == "__main__":
    main()

