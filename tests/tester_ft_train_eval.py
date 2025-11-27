"""
训练过程评估模块：用于在 fine_tuning_1.py 训练过程中评估模型性能。

主要功能：
- 从内存中的 PPOAgent 创建策略（用于训练过程中的评估）
- 在每个训练轮次后评估当前模型性能
- 生成学习曲线和对比表格
- 充分复用 fine_tuning_1.py 和测试框架的代码
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from fine_tuning_1 import PPOAgent
from tests.test_printer import TestPrinter
from tests.test_strategy import (
    PlacementStrategy,
    StrategyContext,
    StrategyResult,
    TestConfig,
    format_config_info,
    run_single_strategy_test,
    SingleTester,
)

__all__ = [
    "InMemoryFineTunedStrategy",
    "create_strategy_from_agent",
    "TrainingEvaluator",
    "evaluate_training_model",
]

#region InMemoryFineTunedStrategy
class InMemoryFineTunedStrategy(PlacementStrategy):
    """
    从内存中的 PPOAgent 创建策略（用于训练过程中的评估）。
    
    与 FineTunedPlacementStrategy 的区别：
    - FineTunedPlacementStrategy: 从文件加载模型
    - InMemoryFineTunedStrategy: 直接使用内存中的 PPOAgent 对象
    
    复用 tester_ft1.py 中的 GreedyPPOAgent 逻辑，但直接从内存中的 agent 创建。
    """

    name: str = "finetuned_in_memory"

    def __init__(
        self,
        agent: PPOAgent,
        *,
        k_hop: int = 1,
        verbose: bool = False,
        model_name: Optional[str] = None,
    ):
        """
        参数:
            agent: PPOAgent 对象（必须已初始化）
            k_hop: BFS 扩展时的 k 跳邻居参数（默认 1）
            verbose: 是否打印详细的放置过程
            model_name: 模型名称（用于日志记录，可选）
        """
        self.agent = agent
        self.k_hop = k_hop
        self.verbose = verbose
        self.model_name = model_name or "in-memory-model"

    def prepare(self, env: SimuVNEEnv) -> None:  # noqa: ARG002
        """确保模型处于评估模式。"""
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
        
        使用贪心策略（按概率从高到低排序），而不是训练时的采样策略。
        这通过创建一个临时的 GreedyPPOAgent 来实现。
        """
        # 创建临时环境副本（避免修改主环境）
        sim_env = copy.deepcopy(env)
        
        # 为了使用贪心策略，我们需要创建一个临时的 GreedyPPOAgent
        # 但为了简化，我们直接使用原 agent，只是确保使用贪心模式
        # 实际上，我们可以通过修改 _generate_priority_lists 的行为来实现
        # 但为了不修改原 agent，我们创建一个包装器
        
        # 直接调用 act，但使用贪心模式（通过修改优先级列表生成方式）
        # 由于 act 方法内部会调用 _generate_priority_lists，我们需要确保使用贪心版本
        # 最简单的方式是创建一个临时的 GreedyPPOAgent 包装器
        
        # 复用 tester_ft1.py 中的 GreedyPPOAgent 逻辑
        from tests.tester_ft1 import GreedyPPOAgent
        
        # 创建贪心版本的 agent（复用策略和价值网络）
        greedy_agent = GreedyPPOAgent(
            self.agent.policy,
            self.agent.value_net,
            device=str(self.agent.device),
        )
        
        mapping, logprob, value = greedy_agent.act(
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
            "model_name": self.model_name,
        }
        
        # 保存调试信息（如果可用）
        debug_info = getattr(greedy_agent, "_last_debug_info", None)
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


def create_strategy_from_agent(
    agent: PPOAgent,
    *,
    k_hop: int = 1,
    verbose: bool = False,
    model_name: Optional[str] = None,
) -> PlacementStrategy:
    """
    从内存中的 PPOAgent 创建策略实例。
    
    参数:
        agent: PPOAgent 对象
        k_hop: BFS k 跳参数
        verbose: 是否详细输出
        model_name: 模型名称（用于日志记录）
    
    返回:
        InMemoryFineTunedStrategy 实例
    """
    return InMemoryFineTunedStrategy(
        agent=agent,
        k_hop=k_hop,
        verbose=verbose,
        model_name=model_name,
    )
#endregion


#region TrainingEvaluator
class TrainingEvaluator:
    """
    训练过程评估器：在训练过程中评估模型性能（仅纵向对比）。
    
    功能：
    - 在每个训练轮次后评估当前模型
    - 记录学习曲线和性能指标
    - 生成评估报告和图表（仅显示模型在不同训练轮次的表现）
    - 充分复用 TestPrinter 和 test_strategy 框架
    """

    def __init__(
        self,
        *,
        eval_config: TestConfig,
        output_dir: Optional[str] = None,
        session_name: Optional[str] = None,
        enable_logging: bool = True,
        enable_plotting: bool = True,
        test_initial_model: bool = True,
        test_pretrained_model: bool = True,
        pretrain_checkpoint: Optional[str] = None,
    ):
        """
        初始化训练评估器。
        
        参数:
            eval_config: 评估配置（TestConfig 对象）
            output_dir: 输出目录（默认: tests/outs/ft_train_eval）
            session_name: 会话名称（默认: 使用时间戳）
            enable_logging: 是否启用日志记录
            enable_plotting: 是否启用图表生成
            test_initial_model: 是否测试预训练前的初始模型（随机初始化，默认: True）
            test_pretrained_model: 是否测试预训练后未经微调的模型（默认: True）
            pretrain_checkpoint: 预训练模型检查点路径（用于测试预训练模型，可选）
        """
        self.eval_config = eval_config
        
        # 设置输出目录
        if output_dir is None:
            base_dir = Path(__file__).resolve().parent / "outs" / "ft_train_eval"
            output_dir = str(base_dir)
        
        self.output_dir = output_dir
        
        # 创建会话目录
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"training_eval_{timestamp}"
        
        self.session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 初始化评估结果存储
        self.evaluation_results: List[Dict[str, Any]] = []
        self.enable_logging = enable_logging
        self.enable_plotting = enable_plotting
        self.test_initial_model = test_initial_model
        self.test_pretrained_model = test_pretrained_model
        self.pretrain_checkpoint = pretrain_checkpoint
        
        # 存储基准模型评估结果
        self.initial_model_result: Optional[Dict[str, Any]] = None
        self.pretrained_model_result: Optional[Dict[str, Any]] = None
        
        # 创建 TestPrinter（用于评估结果输出）
        self.printer = TestPrinter(
            enable_logging=enable_logging,
            enable_plotting=self.enable_plotting,
            output_dir=self.output_dir,
            session_name=session_name,
            test_scope="ft_train_eval",
        )
        
        # 记录评估配置
        if enable_logging:
            self._log_evaluation_config()

    def evaluate_initial_model(self) -> Optional[Dict[str, Any]]:
        """
        评估预训练前的初始模型（随机初始化）。
        
        返回:
            评估结果字典，如果未启用则返回 None
        """
        if not self.test_initial_model:
            return None
        
        from fine_tuning_1 import SimuVNE, ValueNet
        
        print("\n" + "=" * 80)
        print("【评估初始模型】测试预训练前的随机初始化模型")
        print("=" * 80)
        
        # 创建随机初始化的策略和价值网络
        policy = SimuVNE()
        value_net = ValueNet()
        agent = PPOAgent(policy, value_net, device=self.eval_config.device)
        
        # 创建策略
        strategy_factory = lambda: create_strategy_from_agent(
            agent=agent,
            k_hop=1,
            verbose=False,
            model_name="initial_random",
        )
        
        # 准备评估轮次
        table_title = "Initial Model (Random Initialization)"
        config_info = format_config_info(
            self.eval_config,
            workflow_keys=self.eval_config.workflow_types.keys(),
        )
        config_info["model_name"] = "initial_random"
        config_info["model_type"] = "random_initialization"
        
        self.printer.start_round(table_title=table_title, config_info=config_info)
        
        # 评估初始模型
        model_result = run_single_strategy_test(
            strategy_factory=strategy_factory,
            tester_cls=SingleTester,
            config=self.eval_config,
            detail_print=False,
            printer=self.printer,
            strategy_label="initial",
        )
        
        # 完成当前轮次评估
        self.printer.finalize()
        
        # 提取评估结果
        round_data = self.printer._round_logs[-1] if self.printer._round_logs else None
        if round_data:
            model_eval = None
            for result in round_data["results"]:
                if result.get("strategy") == "initial":
                    model_eval = result
                    break
            
            if model_eval:
                self.initial_model_result = {
                    "update_idx": -2,  # 初始模型使用 -2
                    "update_number": 0,  # 初始模型使用 0
                    "timestamp": datetime.now().isoformat(),
                    "model_evaluation": model_eval,
                    "model_type": "initial_random",
                }
                
                # 保存结果
                if self.enable_logging:
                    eval_json_path = os.path.join(
                        self.session_dir,
                        "eval_initial_model.json"
                    )
                    with open(eval_json_path, "w", encoding="utf-8") as f:
                        json.dump(self.initial_model_result, f, indent=2, ensure_ascii=False)
                
                print(f"  初始模型评估结果: 接受率={model_eval.get('acceptance_rate', 0.0)*100:.2f}%, "
                      f"平均r_t={model_eval.get('avg_r_t', 0.0):.3f}, "
                      f"平均跳数={model_eval.get('avg_hops', 0.0):.2f}")
                
                return self.initial_model_result
        
        return None

    def evaluate_pretrained_model(self, pretrain_checkpoint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        评估预训练后未经微调的模型。
        
        参数:
            pretrain_checkpoint: 预训练模型检查点路径（如果为None，使用初始化时提供的路径）
        
        返回:
            评估结果字典，如果未启用或未找到模型则返回 None
        """
        if not self.test_pretrained_model:
            return None
        
        checkpoint_path = pretrain_checkpoint or self.pretrain_checkpoint
        if not checkpoint_path:
            # 尝试自动查找
            from tests.tester_ft_n import find_latest_pretrain_checkpoint
            checkpoint_path = find_latest_pretrain_checkpoint()
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"\n⚠️  未找到预训练模型，跳过预训练模型评估")
            print(f"   检查点路径: {checkpoint_path}")
            return None
        
        from fine_tuning_1 import ValueNet
        from tests.tester_ft_n import load_pretrain_policy
        
        print("\n" + "=" * 80)
        print("【评估预训练模型】测试预训练后未经微调的模型")
        print("=" * 80)
        print(f"  检查点: {os.path.basename(checkpoint_path)}")
        
        try:
            # 加载预训练策略网络
            policy, cfg = load_pretrain_policy(checkpoint_path, self.eval_config.device)
            
            # 创建随机初始化的价值网络（预训练模型没有价值网络）
            value_net = ValueNet(
                input_dim=int(cfg.get("input_dim", 6)),
                hidden_dim=int(cfg.get("hidden_dim", 64)),
            )
            value_net.to(self.eval_config.device)
            value_net.eval()
            
            agent = PPOAgent(policy, value_net, device=self.eval_config.device)
        except Exception as e:
            print(f"  ✗ 加载预训练模型失败: {e}")
            return None
        
        # 创建策略
        strategy_factory = lambda: create_strategy_from_agent(
            agent=agent,
            k_hop=1,
            verbose=False,
            model_name="pretrained",
        )
        
        # 准备评估轮次
        table_title = "Pretrained Model (Before Fine-tuning)"
        config_info = format_config_info(
            self.eval_config,
            workflow_keys=self.eval_config.workflow_types.keys(),
        )
        config_info["model_name"] = "pretrained"
        config_info["model_type"] = "pretrained"
        config_info["checkpoint"] = os.path.basename(checkpoint_path)
        
        self.printer.start_round(table_title=table_title, config_info=config_info)
        
        # 评估预训练模型
        model_result = run_single_strategy_test(
            strategy_factory=strategy_factory,
            tester_cls=SingleTester,
            config=self.eval_config,
            detail_print=False,
            printer=self.printer,
            strategy_label="pretrained",
        )
        
        # 完成当前轮次评估
        self.printer.finalize()
        
        # 提取评估结果
        round_data = self.printer._round_logs[-1] if self.printer._round_logs else None
        if round_data:
            model_eval = None
            for result in round_data["results"]:
                if result.get("strategy") == "pretrained":
                    model_eval = result
                    break
            
            if model_eval:
                self.pretrained_model_result = {
                    "update_idx": -1,  # 预训练模型使用 -1
                    "update_number": 0,  # 预训练模型使用 0（与初始模型区分）
                    "timestamp": datetime.now().isoformat(),
                    "model_evaluation": model_eval,
                    "model_type": "pretrained",
                    "checkpoint": os.path.basename(checkpoint_path),
                }
                
                # 保存结果
                if self.enable_logging:
                    eval_json_path = os.path.join(
                        self.session_dir,
                        "eval_pretrained_model.json"
                    )
                    with open(eval_json_path, "w", encoding="utf-8") as f:
                        json.dump(self.pretrained_model_result, f, indent=2, ensure_ascii=False)
                
                print(f"  预训练模型评估结果: 接受率={model_eval.get('acceptance_rate', 0.0)*100:.2f}%, "
                      f"平均r_t={model_eval.get('avg_r_t', 0.0):.3f}, "
                      f"平均跳数={model_eval.get('avg_hops', 0.0):.2f}")
                
                return self.pretrained_model_result
        
        return None

    def _log_evaluation_config(self) -> None:
        """记录评估配置到日志文件。"""
        if not self.enable_logging or not self.printer._log_file:
            return
        
        self.printer._log_file.write("=" * 80 + "\n")
        self.printer._log_file.write("训练过程评估配置\n")
        self.printer._log_file.write("=" * 80 + "\n")
        self.printer._log_file.write(f"会话目录: {self.printer.session_dir}\n")
        config_info = format_config_info(
            self.eval_config,
            workflow_keys=self.eval_config.workflow_types.keys(),
        )
        self.printer._log_file.write(f"评估配置:\n")
        for key, value in config_info.items():
            self.printer._log_file.write(f"  {key}: {value}\n")
        self.printer._log_file.write(f"\n基准模型测试:\n")
        self.printer._log_file.write(f"  测试初始模型: {self.test_initial_model}\n")
        self.printer._log_file.write(f"  测试预训练模型: {self.test_pretrained_model}\n")
        if self.pretrain_checkpoint:
            self.printer._log_file.write(f"  预训练检查点: {self.pretrain_checkpoint}\n")
        self.printer._log_file.write("=" * 80 + "\n\n")
        self.printer._log_file.flush()

    def evaluate_update(
        self,
        agent: PPOAgent,
        update_idx: int,
        training_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        评估单个训练轮次后的模型性能（仅纵向对比，静默模式）。
        
        参数:
            agent: 当前训练轮次后的 PPOAgent 对象
            update_idx: 训练轮次索引（从0开始）
            training_stats: 训练统计信息（可选，用于记录）
        
        返回:
            评估结果字典
        """
        # 创建策略工厂（从内存中的 agent）
        model_name = f"update_{update_idx + 1}"
        strategy_factory = lambda: create_strategy_from_agent(
            agent=agent,
            k_hop=1,
            verbose=False,
            model_name=model_name,
        )
        
        # 准备评估轮次
        table_title = f"Training Update {update_idx + 1}"
        config_info = format_config_info(
            self.eval_config,
            workflow_keys=self.eval_config.workflow_types.keys(),
        )
        # 添加训练相关信息
        config_info["training_update"] = update_idx + 1
        config_info["model_name"] = model_name
        
        self.printer.start_round(table_title=table_title, config_info=config_info)
        
        # 评估当前模型（静默模式，不打印详细过程）
        model_result = run_single_strategy_test(
            strategy_factory=strategy_factory,
            tester_cls=SingleTester,
            config=self.eval_config,
            detail_print=False,
            printer=self.printer,
            strategy_label="finetuned",
        )
        
        # 完成当前轮次评估
        self.printer.finalize()
        
        # 提取评估结果
        round_data = self.printer._round_logs[-1] if self.printer._round_logs else None
        if round_data:
            # 找到当前模型的评估结果
            model_eval = None
            for result in round_data["results"]:
                if result.get("strategy") == "finetuned":
                    model_eval = result
                    break
            
            eval_result = {
                "update_idx": update_idx,
                "update_number": update_idx + 1,
                "timestamp": datetime.now().isoformat(),
                "model_evaluation": model_eval,
                "training_stats": training_stats,
            }
            
            self.evaluation_results.append(eval_result)
            
            # 保存单轮次评估结果
            if self.enable_logging:
                eval_json_path = os.path.join(
                    self.session_dir,
                    f"eval_update_{update_idx + 1}.json"
                )
                with open(eval_json_path, "w", encoding="utf-8") as f:
                    json.dump(eval_result, f, indent=2, ensure_ascii=False)
            
            return eval_result
        
        return {}

    def finalize(self) -> None:
        """完成评估，生成最终报告和图表。"""
        if not self.evaluation_results:
            return

        self.printer.print_history_table(
            title="训练过程纵向对比结果",
            include_config=True,
            save_to_file=True,
            output_filename="training_history_table.txt",
        )

        if self.enable_plotting:
            self._plot_learning_curves()

        # 关闭 printer，触发日志与图表收尾
        self.printer.close()

        # 保存评估汇总
        summary_path = os.path.join(self.session_dir, "evaluation_summary.json")
        summary_data = {
            "session_dir": self.session_dir,
            "total_updates": len(self.evaluation_results),
            "evaluations": self.evaluation_results,
            "timestamp": datetime.now().isoformat(),
        }
        if self.initial_model_result:
            summary_data["initial_model"] = self.initial_model_result
        if self.pretrained_model_result:
            summary_data["pretrained_model"] = self.pretrained_model_result
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {self.session_dir}")

    def _plot_learning_curves(self) -> None:
        """绘制训练过程的多指标曲线。"""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import warnings

            warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*font.*")
        except ImportError:
            print("警告: matplotlib 未安装，跳过学习曲线绘制。")
            return

        if not self.evaluation_results:
            return

        # 收集所有评估结果（含基准模型）
        all_results = []
        if self.initial_model_result:
            all_results.append(("初始模型", self.initial_model_result))
        if self.pretrained_model_result:
            all_results.append(("预训练模型", self.pretrained_model_result))
        all_results.extend((f"训练轮次 {item['update_number']}", item) for item in self.evaluation_results)

        if not all_results:
            return

        updates: List[float] = []
        metric_accept = []
        metric_rt = []
        metric_hops = []

        for _, result in all_results:
            model_eval = result.get("model_evaluation", {}) or {}
            update_number = result.get("update_number", 0)
            if update_number == 0:
                if result.get("model_type") == "initial_random":
                    updates.append(0.0)
                else:
                    updates.append(0.5)
            else:
                updates.append(float(update_number))
            metric_accept.append(model_eval.get("acceptance_rate", 0.0) * 100.0)
            metric_rt.append(model_eval.get("avg_r_t", 0.0))
            metric_hops.append(model_eval.get("avg_hops", 0.0))

        baseline_updates: List[float] = []
        baseline_accept: List[float] = []
        baseline_rt: List[float] = []
        baseline_hops: List[float] = []
        training_updates: List[float] = []
        training_accept: List[float] = []
        training_rt: List[float] = []
        training_hops: List[float] = []

        for idx, (_, result) in enumerate(all_results):
            if result.get("update_number", 0) == 0:
                baseline_updates.append(updates[idx])
                baseline_accept.append(metric_accept[idx])
                baseline_rt.append(metric_rt[idx])
                baseline_hops.append(metric_hops[idx])
            else:
                training_updates.append(updates[idx])
                training_accept.append(metric_accept[idx])
                training_rt.append(metric_rt[idx])
                training_hops.append(metric_hops[idx])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Training Learning Curves", fontsize=16, fontweight="bold")

        def _scatter_baseline(ax, xs, ys, label):
            if not xs:
                return
            ax.scatter(xs, ys, s=200, marker="*", color="red", label=label, zorder=5)

        # 接受率
        _scatter_baseline(axes[0, 0], baseline_updates, baseline_accept, "Baseline")
        if training_updates:
            axes[0, 0].plot(training_updates, training_accept, "b-o", linewidth=2, markersize=8, label="Finetuned")
        axes[0, 0].set_xlabel("Update Number")
        axes[0, 0].set_ylabel("Acceptance Rate (%)")
        axes[0, 0].set_title("Acceptance Rate")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 平均 r_t
        _scatter_baseline(axes[0, 1], baseline_updates, baseline_rt, "Baseline")
        if training_updates:
            axes[0, 1].plot(training_updates, training_rt, "g-s", linewidth=2, markersize=8, label="Finetuned")
        axes[0, 1].set_xlabel("Update Number")
        axes[0, 1].set_ylabel("Average r_t")
        axes[0, 1].set_title("Average Return")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # 平均跳数
        _scatter_baseline(axes[1, 0], baseline_updates, baseline_hops, "Baseline")
        if training_updates:
            axes[1, 0].plot(training_updates, training_hops, "r-^", linewidth=2, markersize=8, label="Finetuned")
        axes[1, 0].set_xlabel("Update Number")
        axes[1, 0].set_ylabel("Average Hops")
        axes[1, 0].set_title("Average Hops")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # 任务数趋势
        total_tasks_baseline = []
        total_tasks_training = []
        accepted_baseline = []
        accepted_training = []
        for _, result in all_results:
            model_eval = result.get("model_evaluation", {}) or {}
            if result.get("update_number", 0) == 0:
                total_tasks_baseline.append(model_eval.get("tasks", 0))
                accepted_baseline.append(model_eval.get("accepted", 0))
            else:
                total_tasks_training.append(model_eval.get("tasks", 0))
                accepted_training.append(model_eval.get("accepted", 0))

        if baseline_updates:
            axes[1, 1].scatter(baseline_updates, total_tasks_baseline, s=200, marker="*", color="red", label="Baseline Total", zorder=5)
            axes[1, 1].scatter(baseline_updates, accepted_baseline, s=160, marker="P", color="orange", label="Baseline Accepted", zorder=5)
        if training_updates:
            axes[1, 1].plot(training_updates, total_tasks_training, "m-^", linewidth=2, markersize=8, label="Total Tasks")
            axes[1, 1].plot(training_updates, accepted_training, "c-s", linewidth=2, markersize=8, label="Accepted Tasks")
        axes[1, 1].set_xlabel("Update Number")
        axes[1, 1].set_ylabel("Task Count")
        axes[1, 1].set_title("Task Count Trend")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.session_dir, "learning_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 学习曲线图已保存: {plot_path}")
    
#endregion


#region 便捷函数
def evaluate_training_model(
    agent: PPOAgent,
    update_idx: int,
    *,
    eval_config: TestConfig,
    evaluator: Optional[TrainingEvaluator] = None,
    training_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    评估单个训练轮次后的模型性能（便捷函数）。
    
    参数:
        agent: 当前训练轮次后的 PPOAgent 对象
        update_idx: 训练轮次索引
        eval_config: 评估配置
        evaluator: 训练评估器实例（如果为None，会创建新的）
        training_stats: 训练统计信息（可选）
    
    返回:
        评估结果字典
    """
    if evaluator is None:
        evaluator = TrainingEvaluator(
            eval_config=eval_config,
        )
    
    return evaluator.evaluate_update(agent, update_idx, training_stats)
#endregion


#region 接口和演示
def build_arg_parser(
    eval_defaults: Optional[Dict[str, Any]] = None,
    demo_defaults: Optional[Dict[str, Any]] = None,
) -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。
    
    参数:
        eval_defaults: 评估配置默认值字典（可选）
        demo_defaults: 演示配置默认值字典（可选）
    
    返回:
        argparse.ArgumentParser 实例
    """
    # 使用传入的默认值，如果没有则使用内置默认值
    if eval_defaults is None:
        eval_defaults = {
            "sn_topology": "/home/yc2/mrt/a/topo/SN_topology.json",
            "arrival_rate": 0.2,
            "mean_lifetime": 20.0,
            "max_time_steps": 100,
            "seed": 2025,
            "device": "cpu",
            "penalty": -150.0,
        }
    if demo_defaults is None:
        demo_defaults = {
            "pretrain_checkpoint": None,
            "num_updates": 3,
            "output_dir": None,
            "session_name": None,
            "enable_logging": True,
            "enable_plotting": True,
        }
    
    parser = argparse.ArgumentParser(
        description="训练过程评估器：用于在 fine-tuning 训练过程中评估模型性能"
    )
    parser.add_argument(
        "--sn-topology",
        type=str,
        default=eval_defaults["sn_topology"],
        help=f"SN拓扑文件路径（默认: {eval_defaults['sn_topology']}）",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        help="workflow_name=path，可重复指定",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=eval_defaults["arrival_rate"],
        help=f"评估时的任务到达率（默认: {eval_defaults['arrival_rate']}）",
    )
    parser.add_argument(
        "--mean-lifetime",
        type=float,
        default=eval_defaults["mean_lifetime"],
        help=f"评估时的平均生存时间（默认: {eval_defaults['mean_lifetime']}）",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=eval_defaults["max_time_steps"],
        help=f"评估时的最大时间步数（默认: {eval_defaults['max_time_steps']}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=eval_defaults["seed"],
        help=f"评估时的随机种子（默认: {eval_defaults['seed']}）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=eval_defaults["device"],
        help=f"运行设备（默认: {eval_defaults['device']}）",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=eval_defaults["penalty"],
        help=f"映射失败惩罚（默认: {eval_defaults['penalty']}）",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        default=demo_defaults["pretrain_checkpoint"],
        help="预训练模型检查点路径（用于演示，默认: 自动查找）",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=demo_defaults["num_updates"],
        help=f"演示模式：模拟的训练轮次数（默认: {demo_defaults['num_updates']}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=demo_defaults["output_dir"],
        help="输出目录（默认: tests/outs/ft_train_eval）",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=demo_defaults["session_name"],
        help="会话名称（默认: 使用时间戳）",
    )
    parser.add_argument(
        "--disable-plotting",
        action="store_true",
        help="禁用图表生成",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="禁用日志记录",
    )
    parser.add_argument(
        "--disable-initial-model",
        action="store_true",
        help="禁用初始模型测试（默认: 启用）",
    )
    parser.add_argument(
        "--disable-pretrained-model",
        action="store_true",
        help="禁用预训练模型测试（默认: 启用）",
    )
    return parser


def demo_training_evaluation(
    *,
    eval_config: TestConfig,
    pretrain_checkpoint: Optional[str] = None,
    num_updates: int = 3,
    output_dir: Optional[str] = None,
    session_name: Optional[str] = None,
    enable_logging: bool = True,
    enable_plotting: bool = True,
    test_initial_model: bool = True,
    test_pretrained_model: bool = True,
) -> None:
    """
    演示训练评估功能：模拟几个训练轮次并评估。
    
    参数:
        eval_config: 评估配置
        pretrain_checkpoint: 预训练模型检查点路径（可选）
        num_updates: 模拟的训练轮次数
        output_dir: 输出目录
        session_name: 会话名称
        enable_logging: 是否启用日志记录
        enable_plotting: 是否启用图表生成
    """
    from fine_tuning_1 import SimuVNE, ValueNet
    from tests.tester_ft_n import find_latest_pretrain_checkpoint, load_pretrain_policy
    
    print("=" * 80)
    print("训练过程评估演示")
    print("=" * 80)
    
    # 创建评估器
    evaluator = TrainingEvaluator(
        eval_config=eval_config,
        output_dir=output_dir,
        session_name=session_name,
        enable_logging=enable_logging,
        enable_plotting=enable_plotting,
        test_initial_model=test_initial_model,
        test_pretrained_model=test_pretrained_model,
        pretrain_checkpoint=pretrain_checkpoint,
    )
    
    # 评估基准模型（如果启用）
    if evaluator.test_initial_model:
        evaluator.evaluate_initial_model()
    
    if evaluator.test_pretrained_model:
        evaluator.evaluate_pretrained_model(pretrain_checkpoint)
    
    # 初始化策略和价值网络（模拟训练过程）
    print("\n【初始化】创建策略网络和价值网络...")
    policy = SimuVNE()
    
    # 加载预训练模型（如果提供）
    if pretrain_checkpoint:
        checkpoint_path = pretrain_checkpoint
    else:
        checkpoint_path = find_latest_pretrain_checkpoint()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            policy, _ = load_pretrain_policy(checkpoint_path, eval_config.device)
            print(f"  ✓ 加载预训练模型: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"  ⚠ 加载预训练模型失败: {e}，使用随机初始化")
            policy = SimuVNE()
    else:
        print(f"  ✓ 使用随机初始化的策略网络")
    
    value_net = ValueNet()
    agent = PPOAgent(policy, value_net, device=eval_config.device)
    print(f"  ✓ PPO Agent创建完成 (设备: {eval_config.device})")
    
    # 模拟训练过程：每个轮次后评估
    print(f"\n【开始评估】将模拟 {num_updates} 个训练轮次...")
    print("=" * 80)
    
    for update_idx in range(num_updates):
        print(f"\n训练轮次 {update_idx + 1}/{num_updates}")
        print("-" * 80)
        
        # 模拟训练更新（这里只是演示，实际训练应该在 fine_tuning_1.py 中进行）
        # 在实际使用中，这里应该是真实的训练更新后的 agent
        training_stats = {
            "update_idx": update_idx,
            "avg_return": -10.0 + update_idx * 5.0,  # 模拟性能提升
            "avg_accepted": 10 + update_idx * 2,
            "avg_arrived": 20,
        }
        
        # 评估当前模型
        eval_result = evaluator.evaluate_update(
            agent=agent,
            update_idx=update_idx,
            training_stats=training_stats,
        )
        
        if eval_result:
            model_eval = eval_result.get("model_evaluation", {})
            print(f"  评估结果: 接受率={model_eval.get('acceptance_rate', 0.0)*100:.2f}%, "
                  f"平均r_t={model_eval.get('avg_r_t', 0.0):.3f}, "
                  f"平均跳数={model_eval.get('avg_hops', 0.0):.2f}")
    
    # 完成评估，生成报告
    print("\n" + "=" * 80)
    print("【完成评估】生成最终报告...")
    print("=" * 80)
    evaluator.finalize()
    
    print("\n演示完成！")


def main(argv: Optional[List[str]] = None) -> None:
    """
    主函数：演示训练评估功能或提供使用示例。
    
    参数:
        argv: 命令行参数列表（可选，用于测试）
    
    注意：
        - 如需修改默认参数，可直接修改下面的配置字典
        - 命令行参数会覆盖这些默认值
    """
    from tests.test_configs import parse_workflows, DEFAULT_WORKFLOW_TYPES
    
    # ========== 可手动修改的默认参数 ==========
    # 评估配置默认值（可直接修改）
    eval_config_defaults = {
        "sn_topology": "/home/yc2/mrt/a/topo/SN_topology.json",
        "arrival_rate": 0.7,
        "mean_lifetime": 20.0,
        "max_time_steps": 1000,
        "seed": 2025,
        "device": "cpu",
        "penalty": -150.0,
    }
    
    # 演示配置默认值（可直接修改）
    demo_config_defaults = {
        "pretrain_checkpoint": None,  # None 表示自动查找最新预训练模型
        "num_updates": 30,  # 演示模式：模拟的训练轮次数
        "output_dir": None,  # None 表示使用默认目录
        "session_name": None,  # None 表示使用时间戳
        "enable_logging": True,
        "enable_plotting": True,
    }
    # ==========================================
    
    parser = build_arg_parser(
        eval_defaults=eval_config_defaults,
        demo_defaults=demo_config_defaults,
    )
    args = parser.parse_args(argv)
    
    # 解析 workflow
    workflows = parse_workflows(args.workflow) if args.workflow else DEFAULT_WORKFLOW_TYPES
    
    # 创建评估配置（使用命令行参数，如果未提供则使用上面的默认值）
    eval_config = TestConfig(
        sn_topology_path=args.sn_topology,
        workflow_types=workflows,
        arrival_rate=args.arrival_rate,
        mean_lifetime=args.mean_lifetime,
        max_time_steps=args.max_time_steps,
        device=args.device,
        seed=args.seed,
        penalty=args.penalty,
    )
    
    # 运行演示（使用命令行参数，如果未提供则使用上面的默认值）
    demo_training_evaluation(
        eval_config=eval_config,
        pretrain_checkpoint=args.pretrain_checkpoint,
        num_updates=args.num_updates,
        output_dir=args.output_dir,
        session_name=args.session_name,
        enable_logging=not args.disable_logging,
        enable_plotting=not args.disable_plotting,
        test_initial_model=not args.disable_initial_model,
        test_pretrained_model=not args.disable_pretrained_model,
    )


if __name__ == "__main__":
    main()
#endregion
