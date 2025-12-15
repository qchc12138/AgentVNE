#!/usr/bin/env python3
"""
avg_r_t 随轮次变化图绘制脚本

功能：
- 从summary.json读取不同策略的avg_r_t数据
- 绘制avg_r_t随轮次变化的折线图
- 横轴：轮次编号
- 纵轴：avg_r_t值

使用方法：
    直接运行：python3 avg_rt_by_round.py
    修改下方配置区域来控制脚本行为
"""

# ============================================================================
# 配置区域：修改这里的参数来控制脚本行为
# ============================================================================

# 输入文件路径（相对于脚本目录，或使用绝对路径）
INPUT_FILE = "summary.json"

# 策略选择（None表示显示所有策略，或指定列表）
# 注意：策略名称必须与数据文件中的策略名称完全一致
STRATEGIES = None  # None表示显示所有策略
# STRATEGIES = ["ga", "gal-vne", "greedy", "pretrain", "finetuned"]  # 或指定策略列表

# 输出选项
OUTPUT_FILE = None       # None表示自动生成文件名

# 图表样式
FIG_SIZE = (16, 10)       # 图表大小（宽, 高）
DPI = 200                # 分辨率
MARKER_SIZE = 2          # 标记大小

# 线宽设置
LINEWIDTH = 2.0          # 默认线宽（所有策略使用相同线宽）
# 如果希望不同策略使用不同线宽，可以设置策略特定的线宽映射
STRATEGY_LINEWIDTH_MAP = {
    # "ga": 2.0,
    # "gal-vne": 2.5,
    # "greedy": 2.0,
    # "pretrain": 2.5,
    # "finetuned": 2.5,
}

# 字体大小设置
TITLE_FONTSIZE = 36       # 标题字体大小
LABEL_FONTSIZE = 30       # 坐标轴标签字体大小（xlabel, ylabel）
LEGEND_FONTSIZE = 26      # 图例字体大小
TICK_FONTSIZE = 26        # 坐标轴刻度字体大小

# 策略名称映射（用于图表显示）
# 键：数据文件中的策略名称，值：图表中显示的名称
STRATEGY_NAME_MAP = {
    "ga": "GA",
    "gal-vne": "GAL-VNE",
    "greedy": "Greedy",
    "pretrain": "AgentVNE_Pretrain",
    "finetuned": "AgentVNE",
}

# ============================================================================

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("错误：未安装matplotlib，无法绘图")
    exit(1)


def load_summary(json_path: str) -> Dict:
    """从JSON文件加载summary数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_display_name(strategy: str) -> str:
    """获取策略的显示名称（如果存在映射则使用映射，否则使用原名称）"""
    return STRATEGY_NAME_MAP.get(strategy.strip(), strategy.strip())


def extract_avg_rt_data(summary_data: Dict) -> Dict[str, List[float]]:
    """
    从summary数据中提取每个策略的avg_r_t值
    
    Returns:
        {策略名称: [轮次1的avg_r_t, 轮次2的avg_r_t, ...]}
    """
    rounds = summary_data.get("rounds", [])
    strategy_data: Dict[str, List[float]] = {}
    
    for round_data in rounds:
        results = round_data.get("results", [])
        for result in results:
            strategy = result.get("strategy", "")
            avg_r_t = result.get("avg_r_t", 0.0)
            
            if strategy not in strategy_data:
                strategy_data[strategy] = []
            strategy_data[strategy].append(avg_r_t)
    
    return strategy_data


def plot_avg_rt_by_round(
    summary_data: Dict,
    strategies: Optional[List[str]],
    output_path: Optional[str],
) -> None:
    """绘制avg_r_t随轮次变化的折线图"""
    if not HAS_MATPLOTLIB:
        print("错误：未安装matplotlib，无法绘图")
        return
    
    # 提取数据
    strategy_data = extract_avg_rt_data(summary_data)
    
    if not strategy_data:
        print("错误：数据文件中没有策略数据")
        return
    
    # 确定要显示的策略
    if strategies is None:
        strategies_to_plot = sorted(strategy_data.keys())
    else:
        # 去除空格并过滤存在的策略
        strategies_to_plot = [
            s.strip() for s in strategies 
            if s.strip() in strategy_data
        ]
        if not strategies_to_plot:
            print("错误：指定的策略都不存在")
            return
    
    # 确定轮次数量（使用第一个策略的轮次数）
    num_rounds = len(strategy_data[strategies_to_plot[0]])
    round_indices = list(range(1, num_rounds + 1))  # 轮次从1开始
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 设置标题和标签
    ax.set_title("Average r_t by Round", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Round", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Average r_t", fontsize=LABEL_FONTSIZE)
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)
    
    # 绘制每个策略的折线
    for strategy in strategies_to_plot:
        avg_rt_values = strategy_data.get(strategy, [])
        if len(avg_rt_values) != num_rounds:
            print(f"警告：策略 '{strategy}' 的轮次数 ({len(avg_rt_values)}) 与其他策略不一致 ({num_rounds})")
            continue
        
        # 获取显示名称
        display_name = get_display_name(strategy)
        
        # 获取策略特定的线宽（如果配置了），否则使用默认线宽
        strategy_linewidth = STRATEGY_LINEWIDTH_MAP.get(strategy.strip(), LINEWIDTH)
        
        # 绘制折线
        ax.plot(
            round_indices,
            avg_rt_values,
            marker="o",
            label=display_name,
            markersize=MARKER_SIZE,
            linewidth=strategy_linewidth,
        )
    
    # 设置图例和网格
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    
    # 设置x轴范围
    ax.set_xlim(0.5, num_rounds + 0.5)
    
    # 设置x轴刻度为整数
    ax.set_xticks(round_indices)
    
    plt.tight_layout()
    
    # 保存图表
    if output_path is None:
        output_path = "avg_rt_by_round.png"
    
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 图表已保存到: {output_path}")


def main():
    """主函数"""
    # 检查输入文件是否存在
    input_path = Path(INPUT_FILE)
    if not input_path.is_absolute() and not input_path.exists():
        script_dir = Path(__file__).parent
        input_path = script_dir / INPUT_FILE
    
    if not input_path.exists():
        print(f"错误：输入文件不存在: {INPUT_FILE}")
        print(f"  尝试的路径: {input_path}")
        return
    
    # 加载数据
    print(f"正在加载数据: {input_path}")
    try:
        summary_data = load_summary(str(input_path))
    except Exception as e:
        print(f"错误：加载数据失败: {e}")
        return
    
    # 提取策略数据
    strategy_data = extract_avg_rt_data(summary_data)
    available_strategies = sorted(strategy_data.keys())
    
    # 确定要显示的策略
    if STRATEGIES is None:
        strategies_to_plot = available_strategies
    else:
        # 去除空格并过滤存在的策略
        strategies_to_plot = [s.strip() for s in STRATEGIES if s.strip() in available_strategies]
        missing_strategies = [s.strip() for s in STRATEGIES if s.strip() not in available_strategies]
        if missing_strategies:
            print(f"警告：以下策略不存在: {', '.join(missing_strategies)}")
    
    # 打印配置信息
    print(f"\n配置:")
    print(f"  输入文件: {input_path}")
    print(f"  总轮次数: {summary_data.get('total_rounds', 0)}")
    print(f"  可用策略: {', '.join(available_strategies)}")
    if strategies_to_plot:
        display_names = [get_display_name(s) for s in strategies_to_plot]
        print(f"  显示策略: {', '.join(display_names)} ({len(strategies_to_plot)} 个)")
    else:
        print(f"  显示策略: 无（所有指定策略都不存在）")
    print(f"  输出文件: {OUTPUT_FILE if OUTPUT_FILE else '自动生成'}")
    print()
    
    # 绘制图表
    plot_avg_rt_by_round(
        summary_data,
        strategies=STRATEGIES,
        output_path=OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()

