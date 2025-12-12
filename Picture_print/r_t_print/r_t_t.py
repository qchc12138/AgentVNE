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
INPUT_FILE = "round_4_rt_over_time_data.json"

# 时间轴设置
START_TIME_STEP = 600      # 起始时间步（包含）
END_TIME_STEP = 800     # 结束时间步（包含），None表示使用数据文件中的max_time_steps

# 策略选择（None表示显示所有策略，或指定列表）
# 注意：策略名称必须与数据文件中的策略名称完全一致（去除空格）
STRATEGIES = ["ga", "gal-vne", "greedy", "finetuned"]

# 显示选项
USE_ABS = True           # 是否使用绝对值（True: |r_t|, False: r_t）

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


def plot_rt_over_time(
    data: Dict,
    start_time_step: int,
    end_time_step: Optional[int],
    strategies: Optional[List[str]],
    use_abs: bool,
    output_path: Optional[str],
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
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Simulation Time Step", fontsize=LABEL_FONTSIZE)
    ylabel = "Workflow Weighted Average Hops"
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)
    
    # 绘制每个策略的折线
    for strategy in strategies_to_plot:
        time_step_data = strategies_data.get(strategy, [])
        if not time_step_data:
            continue
        
        # 过滤时间步范围
        filtered_data = filter_data_by_time(time_step_data, start_time_step, end_time_step)
        if not filtered_data:
            print(f"警告：策略 '{strategy}' 在时间步范围 [{start_time_step}, {end_time_step}] 内没有数据")
            continue
        
        # 提取时间和r_t值
        time_steps = [d["time_step"] for d in filtered_data]
        r_t_values = [d["r_t"] for d in filtered_data]
        
        # 是否使用绝对值
        if use_abs:
            r_t_values = [abs(r) for r in r_t_values]
        
        # 获取显示名称
        display_name = get_display_name(strategy)
        
        # 获取策略特定的线宽（如果配置了），否则使用默认线宽
        strategy_linewidth = STRATEGY_LINEWIDTH_MAP.get(strategy.strip(), LINEWIDTH)
        
        # 绘制折线
        ax.plot(
            time_steps,
            r_t_values,
            marker="o",
            label=display_name,
            markersize=MARKER_SIZE,
            linewidth=strategy_linewidth,
        )
    
    # 设置图例和网格
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(start_time_step, end_time_step)
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
    )


if __name__ == "__main__":
    main()
