#!/usr/bin/env python3
"""
任务接受率随时间变化图绘制脚本

功能：
- 从CSV文件读取任务接收情况数据
- 计算每个策略的任务接受率（移动平均接受率）
- 对接受率进行平滑处理
- 绘制接受率随时间步变化的折线图（横轴为时间步0-10000）
- 支持控制时间步的起始和结束位置
- 支持选择显示哪些策略
- 支持策略名称映射（显示更友好的名称）

使用方法：
    直接运行：python3 ac_t_2.py
    修改下方配置区域来控制脚本行为
"""

# ============================================================================
# 配置区域：修改这里的参数来控制脚本行为
# ============================================================================

# 输入CSV文件路径（相对于脚本目录，或使用绝对路径）
INPUT_CSV_FILE = "/home/zrz/AgentVNE/AgentVNE/Picture_print/ac_print_3/round_1_task_acceptance.csv"

# 时间步设置
MAX_TIME_STEPS = 10000   # 总时间步数（用于将任务索引映射到时间步）
START_TIME_STEP = 0      # 起始时间步（包含）
END_TIME_STEP = None     # 结束时间步（包含），None表示使用MAX_TIME_STEPS

# 策略选择（None表示显示所有策略，或指定列表）
# 注意：策略名称必须与CSV文件中的列名完全一致（去除空格）
STRATEGIES = ["ga", "gal-vne", "greedy", "finetuned"]

# 数据平滑选项
TIME_WINDOW = 200          # 平滑窗口大小（任务数量）
                          # 当前任务索引的接受率为前TIME_WINDOW个任务接受率的平均
                          # 如果当前任务索引小于TIME_WINDOW，则使用0-当前任务索引的均值

# Y轴范围设置
Y_AXIS_MIN = 0.4          # Y轴最小值
Y_AXIS_MAX = 1.0          # Y轴最大值

# 输出选项
OUTPUT_FILE = None       # None表示自动生成文件名

# 图表样式
FIG_SIZE = (16, 10)       # 图表大小（宽, 高）
DPI = 200                # 分辨率
MARKER_SIZE = 2          # 标记大小

# 线宽设置
LINEWIDTH = 2.0          # 默认线宽（所有策略使用相同线宽）
# 如果希望不同策略使用不同线宽，可以设置策略特定的线宽映射
# 格式：{策略名称: 线宽值}
# 如果策略不在映射中，则使用默认线宽 LINEWIDTH
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
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("错误：未安装matplotlib，无法绘图")
    exit(1)


def load_csv_data(csv_path: str) -> Dict[str, List[int]]:
    """
    从CSV文件加载任务接收情况数据
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        strategy_data: {策略名称: [任务1的接收情况(1/0), 任务2的接收情况(1/0), ...]}
    """
    strategy_data: Dict[str, List[int]] = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # 获取所有策略名称（列名）
        strategy_names = reader.fieldnames
        if strategy_names is None:
            raise ValueError("CSV文件没有列名")
        
        # 初始化策略数据字典
        for strategy in strategy_names:
            strategy_data[strategy] = []
        
        # 读取数据行
        for row in reader:
            for strategy in strategy_names:
                value_str = row.get(strategy, "").strip()
                if value_str:
                    try:
                        value = int(value_str)
                        strategy_data[strategy].append(value)
                    except ValueError:
                        strategy_data[strategy].append(0)  # 无法解析的值记为0
                else:
                    strategy_data[strategy].append(0)  # 空值记为0
    
    return strategy_data


def smooth_acceptance_rates(
    acceptance_list: List[int],
    time_window: int
) -> List[float]:
    """
    对任务接收情况进行平滑处理，计算移动平均接受率
    
    Args:
        acceptance_list: 任务接收情况列表（1=接受，0=拒绝）
        time_window: 平滑窗口大小（任务数量）
    
    Returns:
        平滑后的接受率列表（每个值是前TIME_WINDOW个任务接收情况的平均值）
    """
    if len(acceptance_list) == 0:
        return []
    
    smoothed_rates = []
    
    for i in range(len(acceptance_list)):
        # 计算窗口的起始索引
        window_start_idx = max(0, i - time_window + 1)
        
        # 获取窗口内的任务接收情况（1或0）
        window_acceptances = acceptance_list[window_start_idx:i+1]
        
        # 计算窗口内接收情况的平均值（即接受率）
        if window_acceptances:
            smoothed_rate = sum(window_acceptances) / len(window_acceptances)
        else:
            smoothed_rate = float(acceptance_list[i])
        
        smoothed_rates.append(smoothed_rate)
    
    return smoothed_rates


def get_display_name(strategy: str) -> str:
    """获取策略的显示名称（如果存在映射则使用映射，否则使用原名称）"""
    return STRATEGY_NAME_MAP.get(strategy.strip(), strategy.strip())


def plot_acceptance_rate_over_tasks(
    strategy_data: Dict[str, List[int]],
    start_time_step: int,
    end_time_step: Optional[int],
    strategies: Optional[List[str]],
    output_path: Optional[str],
    time_window: int = 50,
    max_time_steps: int = 10000,
    y_axis_min: float = 0.0,
    y_axis_max: float = 1.05,
) -> None:
    """绘制任务接受率随时间步变化的折线图"""
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
    
    # 确定时间步范围
    if not strategies_to_plot:
        return
    
    # 获取第一个策略的任务数量（所有策略应该有相同的任务数量）
    first_strategy = strategies_to_plot[0]
    total_tasks = len(strategy_data[first_strategy])
    
    if end_time_step is None:
        end_time_step = max_time_steps
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 设置标题和标签
    ax.set_xlabel("Simulation Time Step", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Acceptance Rate", fontsize=LABEL_FONTSIZE)
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)
    
    # Y轴格式化：将接受率（0-1）转换为百分数（0-100），但不显示百分号
    def y_formatter(y, pos):
        if abs(y) < 1e-10:  # 如果y值接近0
            return ''  # 不显示标签
        # 将0-1的值转换为0-100的百分数
        percent_value = y * 100
        # 隐藏100标签
        if abs(percent_value - 100) < 1e-5:
            return ''
        # 如果是整数，显示整数；否则显示1位小数
        # 使用更宽松的整数判断条件（考虑浮点数精度）
        rounded = round(percent_value)
        if abs(percent_value - rounded) < 1e-5:
            return f'{int(rounded)}'
        else:
            return f'{percent_value:.1f}'
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
    
    # 绘制每个策略的折线
    for strategy in strategies_to_plot:
        acceptance_list = strategy_data.get(strategy, [])
        if not acceptance_list:
            continue
        
        # 对任务接收情况进行平滑处理（计算移动平均接受率）
        smoothed_rates = smooth_acceptance_rates(acceptance_list, time_window)
        
        # 将任务索引映射到时间步
        # 假设任务均匀分布在时间步上：time_step = task_idx * max_time_steps / total_tasks
        time_steps = []
        for task_idx in range(len(smoothed_rates)):
            # 将任务索引映射到时间步（线性映射）
            time_step = int(task_idx * max_time_steps / total_tasks) if total_tasks > 0 else 0
            time_steps.append(time_step)
        
        # 过滤时间步范围
        filtered_time_steps = []
        filtered_rates = []
        for i, time_step in enumerate(time_steps):
            if start_time_step <= time_step <= end_time_step:
                filtered_time_steps.append(time_step)
                filtered_rates.append(smoothed_rates[i])
        
        if not filtered_time_steps:
            print(f"警告：策略 '{strategy}' 在时间步范围 [{start_time_step}, {end_time_step}] 内没有数据")
            continue
        
        # 获取显示名称
        display_name = get_display_name(strategy)
        
        # 获取策略特定的线宽（如果配置了），否则使用默认线宽
        strategy_linewidth = STRATEGY_LINEWIDTH_MAP.get(strategy.strip(), LINEWIDTH)
        
        # 绘制折线
        ax.plot(
            filtered_time_steps,
            filtered_rates,
            marker="o",
            label=display_name,
            markersize=MARKER_SIZE,
            linewidth=strategy_linewidth,
        )
    
    # 设置图例和网格
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(start_time_step, end_time_step)
    ax.set_ylim(bottom=y_axis_min, top=y_axis_max)  # 设置y轴范围
    plt.tight_layout()
    
    # 保存图表
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = script_dir / f"acceptance_rate_t{start_time_step}_{end_time_step if end_time_step is not None else 'all'}.png"
    
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
    
    # 加载CSV数据
    print(f"正在加载CSV数据: {input_path}")
    try:
        strategy_data = load_csv_data(str(input_path))
    except Exception as e:
        print(f"错误：加载CSV数据失败: {e}")
        return
    
    # 获取可用的策略列表
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
    
    # 获取任务总数
    if strategies_to_plot:
        total_tasks = len(strategy_data[strategies_to_plot[0]])
    else:
        total_tasks = 0
    
    # 确定结束时间步
    end_time_step = END_TIME_STEP if END_TIME_STEP is not None else MAX_TIME_STEPS
    
    # 打印配置信息
    print(f"\n配置:")
    print(f"  总时间步数: {MAX_TIME_STEPS}")
    print(f"  起始时间步: {START_TIME_STEP}")
    print(f"  结束时间步: {end_time_step if END_TIME_STEP is not None else '自动（使用MAX_TIME_STEPS）'}")
    print(f"  总任务数: {total_tasks}")
    print(f"  可用策略: {', '.join(available_strategies)}")
    if strategies_to_plot:
        display_names = [get_display_name(s) for s in strategies_to_plot]
        print(f"  显示策略: {', '.join(display_names)} ({len(strategies_to_plot)} 个)")
    else:
        print(f"  显示策略: 无（所有指定策略都不存在）")
    print(f"  平滑窗口大小: {TIME_WINDOW} 个任务")
    print(f"  输出文件: {OUTPUT_FILE if OUTPUT_FILE else '自动生成'}")
    print()
    
    # 绘制图表
    plot_acceptance_rate_over_tasks(
        strategy_data=strategy_data,
        start_time_step=START_TIME_STEP,
        end_time_step=END_TIME_STEP,
        strategies=STRATEGIES,
        output_path=OUTPUT_FILE,
        time_window=TIME_WINDOW,
        max_time_steps=MAX_TIME_STEPS,
        y_axis_min=Y_AXIS_MIN,
        y_axis_max=Y_AXIS_MAX,
    )


if __name__ == "__main__":
    main()
