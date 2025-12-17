#!/usr/bin/env python3
"""
acceptance_rate 数据提取脚本

功能：
- 从summary.json读取不同策略的acceptance_rate数据
- 保存为CSV文件，每个策略的数据保存为一列
- 第一列为轮次编号（或arrival_rate）

使用方法：
    直接运行：python3 extract_ac_to_csv.py
    修改下方配置区域来控制脚本行为
"""

# ============================================================================
# 配置区域：修改这里的参数来控制脚本行为
# ============================================================================

# 输入文件路径（相对于脚本目录，或使用绝对路径）
INPUT_FILE = "summary.json"

# 输出CSV文件路径
OUTPUT_CSV_FILE = "acceptance_rate_by_round.csv"

# 策略选择（None表示提取所有策略，或指定列表）
# 注意：策略名称必须与数据文件中的策略名称完全一致
STRATEGIES = None  # None表示提取所有策略
# STRATEGIES = ["ga", "gal-vne", "greedy", "pretrain", "finetuned"]  # 或指定策略列表

# ============================================================================

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_summary(json_path: str) -> Dict:
    """从JSON文件加载summary数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_acceptance_rate_data(summary_data: Dict) -> tuple[List[float], Dict[str, List[float]]]:
    """
    从summary数据中提取每个策略的acceptance_rate值
    
    Returns:
        (round_values, strategy_data)
        - round_values: 轮次对应的arrival_rate值列表（从config中提取）
        - strategy_data: {策略名称: [轮次1的acceptance_rate, 轮次2的acceptance_rate, ...]}
    """
    rounds = summary_data.get("rounds", [])
    strategy_data: Dict[str, List[float]] = {}
    round_values: List[float] = []
    
    for round_data in rounds:
        # 提取arrival_rate作为Round列的值
        config = round_data.get("config", {})
        arrival_rate = config.get("arrival_rate", 0.0)
        round_values.append(arrival_rate)
        
        results = round_data.get("results", [])
        for result in results:
            strategy = result.get("strategy", "")
            acceptance_rate = result.get("acceptance_rate", 0.0)
            
            if strategy not in strategy_data:
                strategy_data[strategy] = []
            strategy_data[strategy].append(acceptance_rate)
    
    return round_values, strategy_data


def save_to_csv(
    round_values: List[float],
    strategy_data: Dict[str, List[float]],
    strategies: Optional[List[str]],
    output_path: str,
) -> None:
    """
    将策略数据保存为CSV文件
    
    CSV格式：
    - 第一列：Round（arrival_rate值）
    - 后续列：每个策略的acceptance_rate值
    """
    # 确定要提取的策略
    if strategies is None:
        strategies_to_extract = sorted(strategy_data.keys())
    else:
        # 去除空格并过滤存在的策略
        strategies_to_extract = [
            s.strip() for s in strategies 
            if s.strip() in strategy_data
        ]
        if not strategies_to_extract:
            print("错误：指定的策略都不存在")
            return
    
    # 确定轮次数量（使用第一个策略的轮次数）
    if not strategies_to_extract:
        print("错误：没有可提取的策略数据")
        return
    
    num_rounds = len(strategy_data[strategies_to_extract[0]])
    
    # 构建CSV表头
    csv_headers = ["Round"] + strategies_to_extract
    
    # 构建CSV数据行
    csv_rows = []
    for i, round_val in enumerate(round_values):
        csv_row = [round_val]
        for strategy in strategies_to_extract:
            acceptance_rates = strategy_data.get(strategy, [])
            if i < len(acceptance_rates):
                csv_row.append(acceptance_rates[i])
            else:
                csv_row.append("")  # 如果数据缺失，留空
        csv_rows.append(csv_row)
    
    # 写入CSV文件
    try:
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig支持Excel打开
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            writer.writerows(csv_rows)
        print(f"✓ CSV文件已保存到: {output_path}")
        print(f"  包含 {len(strategies_to_extract)} 个策略，{num_rounds} 个轮次")
    except Exception as exc:
        print(f"错误：保存CSV文件时出错：{exc}")


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
    round_values, strategy_data = extract_acceptance_rate_data(summary_data)
    available_strategies = sorted(strategy_data.keys())
    
    # 确定要提取的策略
    if STRATEGIES is None:
        strategies_to_extract = available_strategies
    else:
        # 去除空格并过滤存在的策略
        strategies_to_extract = [s.strip() for s in STRATEGIES if s.strip() in available_strategies]
        missing_strategies = [s.strip() for s in STRATEGIES if s.strip() not in available_strategies]
        if missing_strategies:
            print(f"警告：以下策略不存在: {', '.join(missing_strategies)}")
    
    # 打印配置信息
    print(f"\n配置:")
    print(f"  输入文件: {input_path}")
    print(f"  总轮次数: {summary_data.get('total_rounds', 0)}")
    print(f"  Round值（arrival_rate）: {round_values}")
    print(f"  可用策略: {', '.join(available_strategies)}")
    if strategies_to_extract:
        print(f"  提取策略: {', '.join(strategies_to_extract)} ({len(strategies_to_extract)} 个)")
    else:
        print(f"  提取策略: 无（所有指定策略都不存在）")
    print(f"  输出文件: {OUTPUT_CSV_FILE}")
    print()
    
    # 保存为CSV
    save_to_csv(
        round_values,
        strategy_data,
        strategies=STRATEGIES,
        output_path=OUTPUT_CSV_FILE,
    )


if __name__ == "__main__":
    main()

