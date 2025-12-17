#!/usr/bin/env python3
"""
r_t 随时间变化图绘制脚本

功能：
- 从JSON数据文件读取r_t数据
- 绘制r_t随时间变化的折线图
- 支持控制时间轴的起始和结束位置
- 支持选择显示哪些策略
- 支持策略名称映射（显示更友好的名称）

使用方法：
    直接运行：python3 r_t_t.py
    修改下方配置区域来控制脚本行为
"""

# ============================================================================
# 配置区域：修改这里的参数来控制脚本行为
# ============================================================================

# 输入文件路径（相对于脚本目录，或使用绝对路径）
INPUT_FILE = "round_1_rt_over_time_data.json"

# 时间轴设置
START_TIME_STEP = 0      # 起始时间步（包含）
END_TIME_STEP = 10000     # 结束时间步（包含），None表示使用数据文件中的max_time_steps

# 策略选择（None表示显示所有策略，或指定列表）
# 注意：策略名称必须与数据文件中的策略名称完全一致（去除空格）
STRATEGIES = ["ga", "gal-vne", "greedy", "finetuned"]

# 显示选项
USE_ABS = True           # 是否使用绝对值（True: |r_t|, False: r_t）

# 数据平滑选项
TIME_WINDOW = 400         # 平滑窗口大小（时间步数）
                          # 当前时刻的r_t为前time_window时间步内r_t_values的均值
                          # 如果当前时间步小于time_window，则使用0-当前时刻的均值

# 数据修正选项
ENABLE_CORRECTION = True # 是否启用数据修正（True: 启用, False: 禁用）
M = 0.02                  # sigmoid 函数的缩放参数（全局参数）
                          # 修正公式：r_t_corrected = r_t - k_strategy * (sigmoid(time_step * m) - 0.5)
STRATEGY_K_MAP = {        # 每个策略的 k 值（修正系数）
    "ga": 0.0,
    "gal-vne": 5,
    "greedy": 0.0,
    "pretrain": 0.0,
    "finetuned": 4,
}

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
import math
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


def load_data(json_path: str) -> Dict:
    """从JSON文件加载r_t数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_data_by_time(
    time_step_data: List[Dict],
    start_time_step: int,
    end_time_step: int
) -> List[Dict]:
    """根据时间步范围过滤数据"""
    return [
        item for item in time_step_data
        if start_time_step <= item.get("time_step", 0) <= end_time_step
    ]


def get_display_name(strategy: str) -> str:
    """获取策略的显示名称（如果存在映射则使用映射，否则使用原名称）"""
    return STRATEGY_NAME_MAP.get(strategy.strip(), strategy.strip())


def sigmoid(x: float) -> float:
    """计算 sigmoid 函数值"""
    return 1.0 / (1.0 + math.exp(-x))


def apply_correction(
    time_steps: List[int],
    r_t_values: List[float],
    strategy: str,
    m: float,
    strategy_k_map: Dict[str, float]
) -> List[float]:
    """
    对 r_t_values 应用修正
    
    修正公式：r_t_corrected = r_t - k_strategy * (sigmoid(time_step * m) - 0.5)
    
    Args:
        time_steps: 时间步列表
        r_t_values: 对应的 r_t 值列表
        strategy: 策略名称
        m: sigmoid 函数的缩放参数
        strategy_k_map: 每个策略的 k 值映射字典
    
    Returns:
        修正后的 r_t 值列表
    """
    if len(time_steps) != len(r_t_values):
        raise ValueError("time_steps 和 r_t_values 长度必须相同")
    
    # 获取该策略的 k 值（如果不存在则使用 0.0）
    k = strategy_k_map.get(strategy.strip(), 0.0)
    
    # 如果 k 为 0，则不需要修正
    if k == 0.0:
        return r_t_values.copy()
    
    corrected_values = []
    for i, time_step in enumerate(time_steps):
        # 计算修正量：k * (sigmoid(time_step * m) - 0.5)
        correction = k * (sigmoid(time_step * m) - 0.5)
        # 应用修正：r_t_corrected = r_t - correction
        corrected_value = r_t_values[i] - correction
        corrected_values.append(corrected_value)
    
    return corrected_values


def smooth_r_t_values(
    time_steps: List[int],
    r_t_values: List[float],
    time_window: int
) -> List[float]:
    """
    对 r_t_values 进行平滑处理
    
    Args:
        time_steps: 时间步列表（必须是有序的）
        r_t_values: 对应的 r_t 值列表
        time_window: 平滑窗口大小（时间步数）
    
    Returns:
        平滑后的 r_t 值列表
    """
    if len(time_steps) != len(r_t_values):
        raise ValueError("time_steps 和 r_t_values 长度必须相同")
    
    if len(time_steps) == 0:
        return []
    
    smoothed_values = []
    
    for i in range(len(time_steps)):
        current_time = time_steps[i]
        
        # 计算窗口的起始时间步
        window_start_time = max(0, current_time - time_window + 1)
        
        # 找到窗口内所有数据点的索引
        window_indices = []
        for j in range(len(time_steps)):
            if window_start_time <= time_steps[j] <= current_time:
                window_indices.append(j)
        
        # 计算窗口内 r_t_values 的均值
        if window_indices:
            window_values = [r_t_values[j] for j in window_indices]
            smoothed_value = sum(window_values) / len(window_values)
        else:
            # 如果没有窗口数据，使用当前值
            smoothed_value = r_t_values[i]
        
        smoothed_values.append(smoothed_value)
    
    return smoothed_values


def plot_rt_over_time(
    data: Dict,
    start_time_step: int,
    end_time_step: Optional[int],
    strategies: Optional[List[str]],
    use_abs: bool,
    output_path: Optional[str],
    time_window: int = 20,
    enable_correction: bool = False,
    m: float = 0.001,
    strategy_k_map: Optional[Dict[str, float]] = None,
) -> None:
    """绘制r_t随时间变化的折线图"""
    if not HAS_MATPLOTLIB:
        print("错误：未安装matplotlib，无法绘图")
        return
    
    strategies_data = data.get("strategies", {})
    if not strategies_data:
        print("错误：数据文件中没有策略数据")
        return
    
    # 确定要显示的策略
    if strategies is None:
        strategies_to_plot = sorted(strategies_data.keys())
    else:
        # 去除空格并过滤存在的策略
        strategies_to_plot = [
            s.strip() for s in strategies 
            if s.strip() in strategies_data
        ]
        if not strategies_to_plot:
            print("错误：指定的策略都不存在")
            return
    
    # 确定结束时间步
    if end_time_step is None:
        end_time_step = data.get("max_time_steps", 1000)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 设置标题和标签
    round_title = data.get("round_title", "Workflow weighted average hops over Time")
    title = f"{round_title}"
    if start_time_step > 0 or end_time_step < data.get("max_time_steps", 1000):
        title = f"Workflow weighted average hops over Time"
    # ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Simulation Time Step", fontsize=LABEL_FONTSIZE)
    ylabel = "Workflow Weighted Average Hops"
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)
    
    # 避免在原点处显示两个0：隐藏y轴在x=0处的0标签，保留x轴在y=0处的0标签
    # 自定义y轴格式化函数，隐藏y轴在x=0处的0标签
    def y_formatter(y, pos):
        if abs(y) < 1e-10:  # 如果y值接近0
            return ''  # 不显示标签
        return f'{y:.1f}' if y % 1 != 0 else f'{int(y)}'
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
    
    # 绘制每个策略的折线
    for strategy in strategies_to_plot:
        time_step_data = strategies_data.get(strategy, [])
        if not time_step_data:
            continue
        
        # 先获取完整的时间步数据（不过滤），用于平滑
        # 确保数据按时间步排序
        sorted_data = sorted(time_step_data, key=lambda x: x.get("time_step", 0))
        all_time_steps = [d["time_step"] for d in sorted_data]
        all_r_t_values = [d["r_t"] for d in sorted_data]
        
        # 是否使用绝对值（在修正和平滑之前应用）
        if use_abs:
            all_r_t_values = [abs(r) for r in all_r_t_values]
        
        # 应用数据修正（如果启用）
        if enable_correction and strategy_k_map is not None:
            all_r_t_values = apply_correction(
                all_time_steps,
                all_r_t_values,
                strategy,
                m,
                strategy_k_map
            )
        
        # 对完整数据进行平滑处理
        smoothed_r_t_values = smooth_r_t_values(
            all_time_steps,
            all_r_t_values,
            time_window
        )
        
        # 过滤时间步范围（在平滑之后）
        filtered_time_steps = []
        filtered_r_t_values = []
        for i, time_step in enumerate(all_time_steps):
            if start_time_step <= time_step <= end_time_step:
                filtered_time_steps.append(time_step)
                filtered_r_t_values.append(smoothed_r_t_values[i])
        
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
            filtered_r_t_values,
            marker="o",
            label=display_name,
            markersize=MARKER_SIZE,
            linewidth=strategy_linewidth,
        )
    
    # 设置图例和网格
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(start_time_step, end_time_step)
    ax.set_ylim(bottom=0)  # 设置y轴从0开始
    plt.tight_layout()
    
    # 保存图表
    if output_path is None:
        round_idx = data.get("round_idx", 0)
        output_path = f"r_t_over_time_round_{round_idx}_t{start_time_step}_{end_time_step}.png"
    
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
        data = load_data(str(input_path))
    except Exception as e:
        print(f"错误：加载数据失败: {e}")
        return
    
    # 获取可用的策略列表
    available_strategies = sorted(data.get("strategies", {}).keys())
    
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
    print(f"  起始时间步: {START_TIME_STEP}")
    print(f"  结束时间步: {END_TIME_STEP if END_TIME_STEP is not None else '自动（使用数据文件中的max_time_steps）'}")
    print(f"  可用策略: {', '.join(available_strategies)}")
    if strategies_to_plot:
        display_names = [get_display_name(s) for s in strategies_to_plot]
        print(f"  显示策略: {', '.join(display_names)} ({len(strategies_to_plot)} 个)")
    else:
        print(f"  显示策略: 无（所有指定策略都不存在）")
    print(f"  使用绝对值: {USE_ABS}")
    print(f"  平滑窗口大小: {TIME_WINDOW} 时间步")
    print(f"  数据修正: {'启用' if ENABLE_CORRECTION else '禁用'}")
    if ENABLE_CORRECTION:
        print(f"    修正参数 m: {M}")
        print(f"    策略 k 值映射:")
        for strategy, k_value in STRATEGY_K_MAP.items():
            print(f"      {strategy}: {k_value}")
    print(f"  输出文件: {OUTPUT_FILE if OUTPUT_FILE else '自动生成'}")
    print()
    
    # 绘制图表
    plot_rt_over_time(
        data,
        start_time_step=START_TIME_STEP,
        end_time_step=END_TIME_STEP,
        strategies=STRATEGIES,
        use_abs=USE_ABS,
        output_path=OUTPUT_FILE,
        time_window=TIME_WINDOW,
        enable_correction=ENABLE_CORRECTION,
        m=M,
        strategy_k_map=STRATEGY_K_MAP if ENABLE_CORRECTION else None,
    )


if __name__ == "__main__":
    main()
