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

__all__ = [
    "FineTunedPlacementStrategy",
    "FT1PlacementStrategy",
    "find_latest_finetune_run",
    "resolve_checkpoint",
    "extract_model_name",
    "run_finetuned_strategy_test",
    "smoke_test_finetuned_strategy",
    "finetuned_strategy_factory",
    "FINETUNED_TESTER_PRESETS",
    "get_finetuned_tester_registry",
]

#region 常量定义
DEFAULT_FINETUNE_OUTPUT_DIR = "/home/yc2/mrt/a/finetuning_putput"
FINETUNED_MODEL_NAME = "model_1"
MODEL_FACTORIES = {
    FINETUNED_MODEL_NAME: SimuVNEModel1,
}

# tester.py 对比实验中可直接引用的 FineTuned 预设
# 默认自动在 model_1 输出目录中查找最新 run
FINETUNED_TESTER_PRESETS: Dict[str, Dict[str, Any]] = {
    "finetuned_model1_latest": {
        "description": "model_1 最新 run（默认输出目录）",
        "policy_path": None,
        "value_path": None,
        "device": "cpu",
        "k_hop": 1,
        "finetune_output_dir": DEFAULT_FINETUNE_OUTPUT_DIR,
    },
}
#endregion


#region tester.py 集成
def _build_finetuned_spec(preset: Dict[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "policy_path": preset.get("policy_path"),
        "value_path": preset.get("value_path"),
        "device": preset.get("device", "cpu"),
        "k_hop": preset.get("k_hop", 1),
        "finetune_output_dir": preset.get("finetune_output_dir", DEFAULT_FINETUNE_OUTPUT_DIR),
    }
    return spec


def get_finetuned_tester_registry() -> Dict[str, Callable[[], PlacementStrategy]]:
    """
    为 tester.py 提供预设的 finetuned 策略工厂映射。
    """
    registry: Dict[str, Callable[[], PlacementStrategy]] = {}
    for label, preset in FINETUNED_TESTER_PRESETS.items():
        spec = _build_finetuned_spec(preset)

        def _factory(spec: Dict[str, Any] = spec) -> PlacementStrategy:
            return finetuned_strategy_factory(**spec)

        registry[label] = _factory
    return registry
#endregion


#region 辅助函数
def find_latest_finetune_run(output_dir: str = DEFAULT_FINETUNE_OUTPUT_DIR) -> Optional[str]:
    """
    查找 fine_tuning_1.py 最新训练结果目录。
    
    fine_tuning_1.py 的 save_training_results 函数会将结果保存到：
    {output_dir}/run_{timestamp}/
    
    此函数会扫描 output_dir 下所有 run_* 目录，返回时间戳最新的那个目录路径。
    
    参数:
        output_dir: fine_tuning 输出根目录，默认为 DEFAULT_FINETUNE_OUTPUT_DIR
    
    返回:
        最新训练结果目录的完整路径（例如：/path/to/finetuning_putput/run_20251121_164318），
        如果未找到则返回 None
    """
    if not os.path.isdir(output_dir):
        return None
    
    # 查找所有 run_* 目录
    run_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("run_"):
            run_dirs.append((item_path, item))
    
    if not run_dirs:
        return None
    
    # 按目录名（时间戳）排序，取最新的
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    return run_dirs[0][0]


def resolve_checkpoint(
    path: Optional[str], *, filename: str, output_dir: str
) -> Optional[str]:
    """
    解析模型检查点路径。
    
    如果 path 为 None，自动查找最新训练结果目录中的文件。
    如果 path 是目录，自动拼接 filename。
    如果 path 是文件，直接返回。
    
    参数:
        path: 模型文件路径、目录路径或 None
        filename: 当 path 是目录或 None 时，要查找的文件名（如 "policy_network.pth"）
        output_dir: fine_tuning 输出根目录（仅在 path=None 时使用）
    
    返回:
        规范化后的完整文件路径，如果未找到则返回 None
    """
    if path:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            candidate = os.path.join(path, filename)
            return candidate if os.path.isfile(candidate) else None
        return path if os.path.isfile(path) else None
    
    # 自动查找最新训练结果
    latest = find_latest_finetune_run(output_dir)
    if not latest:
        return None
    candidate = os.path.join(latest, filename)
    return candidate if os.path.isfile(candidate) else None


def extract_model_name(model_path: Optional[str]) -> str:
    """
    从模型路径中提取模型名称。
    
    优先提取 run_xxx 目录名作为模型名称，如果没有则使用文件名（不含扩展名）。
    如果 model_path 为 None，返回 "unknown"。
    
    参数:
        model_path: 模型文件的完整路径
    
    返回:
        模型名称字符串（例如："run_20251121_164318" 或 "policy_network"）
    """
    if not model_path:
        return "unknown"
    
    # 标准化路径
    normalized_path = os.path.normpath(model_path)
    dirname = os.path.dirname(normalized_path)
    basename = os.path.basename(normalized_path)
    
    # 检查父目录是否是 run_xxx 格式
    parent_dir = os.path.basename(dirname)
    if parent_dir.startswith("run_"):
        return parent_dir
    
    # 否则使用文件名（不含扩展名）
    name_without_ext = os.path.splitext(basename)[0]
    return name_without_ext


def load_policy(path: str, device: str) -> tuple[Any, Dict[str, object]]:
    """
    加载 fine-tuned 策略网络模型。
    
    参数:
        path: 策略网络检查点文件路径
        device: 目标设备（"cpu" 或 "cuda"）
    
    返回:
        (policy, policy_cfg): 加载的策略网络实例和模型配置字典
    
    异常:
        FileNotFoundError: 如果检查点文件不存在
        RuntimeError: 如果加载失败（如 state_dict 不匹配）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到策略模型文件: {path}")
    
    factory = MODEL_FACTORIES["model_1"]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    
    model = factory(
        input_dim=model_cfg.get("input_dim", 6),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        hist_dim=model_cfg.get("hist_dim", 32),
        num_nodes_j=model_cfg.get("num_nodes_j", 10),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, model_cfg


def load_value_net(
    path: str, device: str, policy_cfg: Dict[str, object]
) -> ValueNet:
    """
    加载 fine-tuned 价值网络模型。
    
    参数:
        path: 价值网络检查点文件路径
        device: 目标设备
        policy_cfg: 策略网络的配置参数（用于确定价值网络的输入维度）
    
    返回:
        加载的价值网络实例
    
    异常:
        FileNotFoundError: 如果检查点文件不存在
        RuntimeError: 如果加载失败
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到价值网络文件: {path}")
    
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    value_net = ValueNet(
        input_dim=int(policy_cfg.get("input_dim", 6)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 64)),
    )
    value_net.load_state_dict(state_dict, strict=True)
    value_net.to(device)
    value_net.eval()
    return value_net
#endregion


#region GreedyPPOAgent
class GreedyPPOAgent(PPOAgent):
    """
    贪心版本的 PPOAgent，用于测试时使用。
    
    与原始 PPOAgent 的区别：
    - 重写 _generate_priority_lists 方法，直接按概率从高到低排序，不使用采样
    - 这样可以更好地释放模型性能，因为测试时不需要探索
    """
    
    def _generate_priority_lists(self, probs_matrix: torch.Tensor) -> List[List[int]]:
        """
        从概率矩阵生成优先级列表（贪心方式：直接按概率从高到低排序）。
        
        与训练时的采样方式不同，测试时直接选择概率最高的节点，以获得更好的性能。
        
        参数:
            probs_matrix: 概率矩阵 [N_v, N_s]
        
        返回:
            priority_lists: 每个VN节点对应的SN节点优先级列表（按概率从高到低）
        """
        priority_lists = []
        for i in range(probs_matrix.size(0)):
            _, indices = torch.sort(probs_matrix[i], descending=True)
            priority_lists.append(indices.tolist())
        return priority_lists
#endregion


#region FineTunedPlacementStrategy
@dataclass
class FineTunedPlacementStrategy(PlacementStrategy):
    """
    基于 fine_tuning_1.py 训练得到的 PPO 模型的放置策略。

    仅在内部副本环境上执行搜索与资源扣减，主环境资源更新由测试框架负责。
    该策略加载训练完成的策略网络和价值网络，通过 PPOAgent 的 act 方法
    执行基于优先级列表和 BFS 扩展的放置决策。
    """

    name: str = "finetuned"
    policy_path: Optional[str] = None
    value_path: Optional[str] = None
    device: str = "cpu"
    k_hop: int = 1
    verbose: bool = False
    finetune_output_dir: str = DEFAULT_FINETUNE_OUTPUT_DIR

    def __post_init__(self):
        """初始化时解析模型文件路径。"""
        self._policy_file = resolve_checkpoint(
            self.policy_path,
            filename="policy_network.pth",
            output_dir=self.finetune_output_dir,
        )
        self._value_file = resolve_checkpoint(
            self.value_path,
            filename="value_network.pth",
            output_dir=self.finetune_output_dir,
        )
        self.agent: Optional[PPOAgent] = None
        # 提取模型名称用于日志记录
        self.model_name_display = extract_model_name(self._policy_file)

    def prepare(self, env: SimuVNEEnv) -> None:
        """
        加载策略网络和价值网络，初始化 PPOAgent。
        
        异常:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
        """
        if not self._policy_file or not self._value_file:
            raise FileNotFoundError(
                "未找到 fine-tuned 模型，请先训练或指定 --policy-path/--value-path。"
                f"策略网络: {self._policy_file}, 价值网络: {self._value_file}"
            )
        
        # 打印模型加载信息
        print(f"[FineTuned] 加载模型: {self.model_name_display} ({FINETUNED_MODEL_NAME})")
        print(f"  策略网络: {os.path.basename(self._policy_file)}")
        print(f"  价值网络: {os.path.basename(self._value_file)}")
        
        try:
            policy, cfg = load_policy(self._policy_file, self.device)
            value_net = load_value_net(self._value_file, self.device, cfg)
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
                "FineTunedPlacementStrategy 尚未准备，请先调用 prepare()。"
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


class FT1PlacementStrategy(FineTunedPlacementStrategy):
    """
    无需显式参数的预设策略，自动从默认目录选择最新模型。
    """

    def __init__(
        self,
        *,
        policy_path: Optional[str] = None,
        value_path: Optional[str] = None,
        device: str = "cpu",
        k_hop: int = 1,
        verbose: bool = False,
        finetune_output_dir: str = DEFAULT_FINETUNE_OUTPUT_DIR,
    ):
        super().__init__(
            policy_path=policy_path,
            value_path=value_path,
            device=device,
            k_hop=k_hop,
            verbose=verbose,
            finetune_output_dir=finetune_output_dir,
        )


#region 工厂函数
def finetuned_strategy_factory(
    *,
    policy_path: Optional[str] = None,
    value_path: Optional[str] = None,
    device: str = "cpu",
    k_hop: int = 1,
    verbose: bool = False,
    finetune_output_dir: str = DEFAULT_FINETUNE_OUTPUT_DIR,
) -> PlacementStrategy:
    """
    创建 FineTunedPlacementStrategy 实例的工厂函数。
    
    默认情况下，如果 policy_path 为 None，会自动查找最新训练结果。
    如需指定特定模型，可传入 policy_path 或 value_path。
    
    参数:
        policy_path: 策略网络路径（None 时自动查找最新训练结果）
        value_path: 价值网络路径（None 时在 policy_path 同目录查找）
        device: 运行设备
        k_hop: BFS k 跳参数
        verbose: 是否详细输出
        finetune_output_dir: fine_tuning 输出根目录
    
    返回:
        FineTunedPlacementStrategy 实例
    """
    return FineTunedPlacementStrategy(
        policy_path=policy_path,
        value_path=value_path,
        device=device,
        k_hop=k_hop,
        verbose=verbose,
        finetune_output_dir=finetune_output_dir,
    )
#endregion


#region 运行封装
def run_finetuned_strategy_test(
    *,
    detail_print: bool = False,
    tester_cls: Type[SingleTester] = SingleTester,
    config: Optional[TestConfig] = None,
    config_overrides: Optional[Dict[str, object]] = None,
    printer: Optional[TestPrinter] = None,
    strategy_label: str = "finetuned",
    policy_path: Optional[str] = None,
    value_path: Optional[str] = None,
    device: str = "cpu",
    k_hop: int = 1,
    finetune_output_dir: str = DEFAULT_FINETUNE_OUTPUT_DIR,
):
    """
    统一封装 FineTuned 策略的单次测试。
    
    参数:
        detail_print: 是否打印详细过程
        tester_cls: Tester 类（默认 SingleTester）
        config: 测试配置（可选）
        config_overrides: 配置覆盖字典（可选）
        printer: TestPrinter 实例（可选）
        strategy_label: 策略标签（默认 "finetuned"）
        policy_path: 策略网络路径（None 时自动查找最新训练结果）
        value_path: 价值网络路径（None 时在 policy_path 同目录查找）
        device: 运行设备
        k_hop: BFS k 跳参数
        finetune_output_dir: fine_tuning 输出根目录
    
    返回:
        测试结果字典（包含 tasks、summary、lag_time 等）
    
    异常:
        FileNotFoundError: 如果未找到模型文件
    """
    factory = lambda: finetuned_strategy_factory(
        policy_path=policy_path,
        value_path=value_path,
        device=device,
        k_hop=k_hop,
        verbose=False,
        finetune_output_dir=finetune_output_dir,
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


def smoke_test_finetuned_strategy(
    *,
    detail_print: bool = False,
    policy_path: Optional[str] = None,
    value_path: Optional[str] = None,
    device: str = "cpu",
    k_hop: int = 1,
    finetune_output_dir: str = DEFAULT_FINETUNE_OUTPUT_DIR,
) -> Optional[Dict[str, Any]]:
    """
    较小参数的 FineTuned 冒烟测试，输出详细日志。
    
    如果找不到模型文件，将打印错误信息并返回 None。
    
    参数:
        detail_print: 是否打印详细过程
        policy_path: 策略网络路径（None 时自动查找最新训练结果）
        value_path: 价值网络路径（None 时在 policy_path 同目录查找）
        device: 运行设备
        k_hop: BFS k 跳参数
        finetune_output_dir: fine_tuning 输出根目录
    
    返回:
        测试结果字典，如果未找到模型则返回 None
    """
    # 尝试创建策略实例以检查模型文件是否存在
    strategy = FineTunedPlacementStrategy(
        policy_path=policy_path,
        value_path=value_path,
        device=device,
        k_hop=k_hop,
        verbose=detail_print,
        finetune_output_dir=finetune_output_dir,
    )
    
    if not strategy._policy_file or not strategy._value_file:
        print("[FineTuned] ❌ 未找到模型文件，已跳过测试。")
        print(f"[FineTuned] 策略网络: {strategy._policy_file}")
        print(f"[FineTuned] 价值网络: {strategy._value_file}")
        print(f"[FineTuned] 请确保已运行 fine_tuning_1.py 进行训练，或手动指定 --policy-path/--value-path。")
        return None

    config = get_smoke_config()

    printer = TestPrinter(
        enable_logging=True,
        enable_plotting=False,
        test_scope="finetuned_single",
    )
    try:
        printer.start_round(
            table_title="FineTuned Strategy Smoke Test",
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
                "strategy_name": "FineTunedPlacementStrategy",
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
    parser = argparse.ArgumentParser(description="FineTuned 策略冒烟测试")
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="策略网络路径（policy_network.pth 或其所在目录，默认：自动查找最新训练结果）",
    )
    parser.add_argument(
        "--value-path",
        type=str,
        default=None,
        help="价值网络路径（value_network.pth 或其所在目录，默认：在 policy_path 同目录查找）",
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
        "--finetune-output-dir",
        type=str,
        default=DEFAULT_FINETUNE_OUTPUT_DIR,
        help="fine_tuning 输出根目录",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="是否打印详细日志",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """
    主函数：运行 FineTuned 策略的冒烟测试。
    
    参数:
        argv: 命令行参数列表（可选，用于测试）
    """
    args = build_arg_parser().parse_args(argv)
    result = smoke_test_finetuned_strategy(
        detail_print=args.detail,
        policy_path=args.policy_path,
        value_path=args.value_path,
        device=args.device,
        k_hop=args.k_hop,
        finetune_output_dir=args.finetune_output_dir,
    )
    
    if result is not None:
        summary = result["summary"]
        print(
            "[FineTunedSmoke] total_tasks={total}, accepted={accepted}, acceptance_rate={rate:.2f}%".format(
                total=summary["total_tasks"],
                accepted=summary["accepted"],
                rate=summary["acceptance_rate"] * 100.0,
            )
        )
    else:
        print("[FineTunedSmoke] 测试已取消：未找到模型文件。")
#endregion


if __name__ == "__main__":
    main()
