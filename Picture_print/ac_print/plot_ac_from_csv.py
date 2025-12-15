#!/usr/bin/env python3
"""
acceptance_rate 柱状图绘制脚本（从CSV文件读取）

功能：
- 从CSV文件读取不同策略的acceptance_rate数据
- 绘制acceptance_rate随轮次变化的柱状图
- 横轴：Round（arrival_rate值）
- 纵轴：acceptance_rate值（百分比）

使用方法：
    直接运行：python3 plot_ac_from_csv.py
    修改下方配置区域来控制脚本行为
"""

# ============================================================================
# 配置区域：修改这里的参数来控制脚本行为
# ============================================================================

# 输入CSV文件路径（相对于脚本目录，或使用绝对路径）
# INPUT_CSV_FILE = "acceptance_rate_by_round.csv"
INPUT_CSV_FILE = "fix.csv"

# 策略选择（None表示显示所有策略，或指定列表）
# 注意：策略名称必须与CSV文件中的列名完全一致
# STRATEGIES = None  # None表示显示所有策略
STRATEGIES = ["ga", "gal-vne", "greedy", "finetuned"]  # 或指定策略列表

# Y轴范围设置
# 如果设置为None，则自动根据数据范围调整
# 如果设置为具体值，则使用指定的范围
Y_AXIS_MIN = 50         # Y轴最小值（None表示自动）
Y_AXIS_MAX = 100         # Y轴最大值（None表示自动）

# 输出选项
OUTPUT_FILE = None       # None表示自动生成文件名

# 图表样式
FIG_SIZE = (16, 10)       # 图表大小（宽, 高）
DPI = 200                # 分辨率
BAR_WIDTH = 0.15          # 柱状图宽度（每个策略的柱子宽度）
BAR_GAP = 0.05            # 柱子之间的间距

# 线宽设置
LINEWIDTH = 3         # 默认线宽（所有策略使用相同线宽）
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
LABEL_FONTSIZE = 34       # 坐标轴标签字体大小（xlabel, ylabel）
LEGEND_FONTSIZE = 32      # 图例字体大小
TICK_FONTSIZE = 34        # 坐标轴刻度字体大小

# 策略名称映射（用于图表显示）
# 键：CSV文件中的策略名称，值：图表中显示的名称
STRATEGY_NAME_MAP = {
    "ga": "GA",
    "gal-vne": "GAL-VNE",
    "greedy": "Greedy",
    "pretrain": "AgentVNE_Pretrain",
    "finetuned": "AgentVNE",
}

# ============================================================================

import csv
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("错误：未安装matplotlib，无法绘图")
    exit(1)


def load_csv_data(csv_path: str) -> tuple[List[float], Dict[str, List[float]]]:
    """
    从CSV文件加载数据
    
    Returns:
        (round_values, strategy_data)
        - round_values: Round列的浮点值列表（直接使用CSV中的值）
        - strategy_data: {策略名称: [轮次1的avg_r_t, 轮次2的avg_r_t, ...]}
    """
    round_values: List[float] = []
    strategy_data: Dict[str, List[float]] = {}
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        # 获取所有列名（除了Round列）
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV文件没有列名")
        
        strategy_columns = [col for col in fieldnames if col != "Round"]
        
        # 初始化策略数据字典
        for strategy in strategy_columns:
            strategy_data[strategy] = []
        
        # 读取数据行
        for row in reader:
            round_str = row.get("Round", "").strip()
            if not round_str:
                continue
            
            # 将Round列转换为浮点数（直接使用CSV中的值）
            try:
                round_value = float(round_str)
            except (ValueError, TypeError):
                # 如果无法转换，跳过这一行
                continue
            
            round_values.append(round_value)
            
            for strategy in strategy_columns:
                value_str = row.get(strategy, "").strip()
                if value_str:
                    try:
                        value = float(value_str)
                        # 如果值小于等于1，认为是小数形式，需要转换为百分比
                        # 如果值大于1，认为已经是百分比形式
                        if value <= 1.0:
                            value = value * 100.0
                        strategy_data[strategy].append(value)
                    except ValueError:
                        strategy_data[strategy].append(None)  # 无法解析的值
                else:
                    strategy_data[strategy].append(None)  # 空值
    
    return round_values, strategy_data


def get_display_name(strategy: str) -> str:
    """获取策略的显示名称（如果存在映射则使用映射，否则使用原名称）"""
    return STRATEGY_NAME_MAP.get(strategy.strip(), strategy.strip())


def plot_acceptance_rate_by_round(
    round_values: List[float],
    strategy_data: Dict[str, List[float]],
    strategies: Optional[List[str]],
    output_path: Optional[str],
) -> None:
    """绘制acceptance_rate随轮次变化的柱状图"""
    if not HAS_MATPLOTLIB:
        print("错误：未安装matplotlib，无法绘图")
        return
    
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
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 设置标题和标签
    # ax.set_title("Acceptance Rate by Round", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Arrival Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Acceptance Rate (%)", fontsize=LABEL_FONTSIZE)
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)
    
    # 准备柱状图数据
    num_strategies = len(strategies_to_plot)
    num_rounds = len(round_values)
    
    # 计算每个策略的柱子位置
    x_positions = []
    for i, round_val in enumerate(round_values):
        # 为每个round计算策略柱子的x位置
        base_pos = i
        positions = []
        total_width = num_strategies * BAR_WIDTH + (num_strategies - 1) * BAR_GAP
        start_offset = -total_width / 2 + BAR_WIDTH / 2
        
        for j in range(num_strategies):
            positions.append(base_pos + start_offset + j * (BAR_WIDTH + BAR_GAP))
        x_positions.append(positions)
    
    # 绘制每个策略的柱状图
    for strategy_idx, strategy in enumerate(strategies_to_plot):
        acceptance_rates = strategy_data.get(strategy, [])
        
        # 提取有效数据
        valid_values = []
        valid_x_positions = []
        for i, (round_val, value) in enumerate(zip(round_values, acceptance_rates)):
            if value is not None:
                valid_values.append(value)  # acceptance_rate已经是百分比格式
                valid_x_positions.append(x_positions[i][strategy_idx])
        
        if not valid_values:
            print(f"警告：策略 '{strategy}' 没有有效数据")
            continue
        
        # 获取显示名称
        display_name = get_display_name(strategy)
        
        # 绘制柱状图
        ax.bar(
            valid_x_positions,
            valid_values,
            width=BAR_WIDTH,
            label=display_name,
            alpha=0.8,
        )
    
    # 设置图例和网格
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    
    # 设置x轴
    if round_values:
        # x轴刻度位置为每个round的中心位置
        ax.set_xticks(range(len(round_values)))
        # x轴刻度标签为Round列的值
        ax.set_xticklabels([f"{val:.2f}" for val in round_values])
        # 设置x轴范围
        ax.set_xlim(-0.5, len(round_values) - 0.5)
    
    # 设置y轴范围
    if Y_AXIS_MIN is not None and Y_AXIS_MAX is not None:
        # 手动设置的范围
        ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    else:
        # 自适应设置y轴范围
        # 收集所有有效数据值
        all_values = []
        for strategy in strategies_to_plot:
            acceptance_rates = strategy_data.get(strategy, [])
            for value in acceptance_rates:
                if value is not None:
                    all_values.append(value)
        
        if all_values:
            min_value = min(all_values)
            max_value = max(all_values)
            # 添加一些边距（5%的边距）
            value_range = max_value - min_value
            margin = max(value_range * 0.05, 1.0)  # 至少1%的边距
            
            # 如果只设置了最小值或最大值，使用数据计算另一个值
            if Y_AXIS_MIN is not None:
                y_min = Y_AXIS_MIN
                y_max = min(105, max_value + margin)
            elif Y_AXIS_MAX is not None:
                y_min = max(0, min_value - margin)
                y_max = Y_AXIS_MAX
            else:
                # 完全自动
                y_min = max(0, min_value - margin)  # 最小值不低于0
                y_max = min(105, max_value + margin)  # 最大值不超过105（acceptance_rate是百分比）
            
            ax.set_ylim(y_min, y_max)
        else:
            # 如果没有有效数据，使用默认范围或手动设置的范围
            if Y_AXIS_MIN is not None and Y_AXIS_MAX is not None:
                ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
            else:
                ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # 保存图表
    if output_path is None:
        output_path = "acceptance_rate_by_round.png"
    
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 图表已保存到: {output_path}")


def main():
    """主函数"""
    # 检查输入文件是否存在
    input_path = Path(INPUT_CSV_FILE)
    if not input_path.is_absolute() and not input_path.exists():
        script_dir = Path(__file__).parent
        input_path = script_dir / INPUT_CSV_FILE
    
    if not input_path.exists():
        print(f"错误：输入文件不存在: {INPUT_CSV_FILE}")
        print(f"  尝试的路径: {input_path}")
        return
    
    # 加载数据
    print(f"正在加载数据: {input_path}")
    try:
        round_values, strategy_data = load_csv_data(str(input_path))
    except Exception as e:
        print(f"错误：加载数据失败: {e}")
        return
    
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
    print(f"  总轮次数: {len(round_values)}")
    print(f"  Round值: {round_values}")
    print(f"  可用策略: {', '.join(available_strategies)}")
    if strategies_to_plot:
        display_names = [get_display_name(s) for s in strategies_to_plot]
        print(f"  显示策略: {', '.join(display_names)} ({len(strategies_to_plot)} 个)")
    else:
        print(f"  显示策略: 无（所有指定策略都不存在）")
    print(f"  Y轴范围: ", end="")
    if Y_AXIS_MIN is not None and Y_AXIS_MAX is not None:
        print(f"手动设置 [{Y_AXIS_MIN}, {Y_AXIS_MAX}]")
    else:
        print("自动调整")
    print(f"  输出文件: {OUTPUT_FILE if OUTPUT_FILE else '自动生成'}")
    print()
    
    # 绘制图表
    plot_acceptance_rate_by_round(
        round_values,
        strategy_data,
        strategies=STRATEGIES,
        output_path=OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()

