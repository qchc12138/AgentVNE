"""workflow1执行器（带资源监控） - 可配置的工作流执行和监控程序"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent_flows.workflows.workflow1 import run_booking_workflow
from src.agent_flows.observability.node_wrapper import TracedWorkflowRunner


# ==================== 配置区域 ====================

# 监控配置：选择要统计的指标
MONITORING_CONFIG = {
    'enable_cpu': True,        # CPU使用率
    'enable_memory': True,     # 内存使用
    'enable_storage': True,    # 存储IO
    'enable_gpu': True,        # GPU使用（如果可用）
    'enable_network': True,    # 网络通信
}

# 工作流配置
WORKFLOW_NAME = "workflow1"  # 工作流名称（用于结果文件命名）
WORKFLOW_MODULE = "src.agent_flows.workflows.workflow1"  # 工作流模块路径
USER_INPUT = "请帮我在和平路搜索酒店，预算在200到300元之间"  # 用户输入

# 节点配置：定义工作流中的节点名称到函数名的映射
# 这样就能追踪每个节点的资源消耗
# 格式：{'显示名称': '函数名'}
NODE_FUNCTIONS = {
    '节点1-意图理解': 'intent_understanding_node',
    '节点2-参数提取': 'parameter_extraction_node',
    '节点3-搜索规划': 'search_planning_node',
    '节点4-酒店搜索': 'hotel_search_node',
    '节点5-筛选选择': 'filter_and_selection_node',
    '节点6-支付': 'payment_node',
    '节点7-总结': 'summarization_node',
}

# 输出配置
OUTPUT_DIR = "results/workflow1"  # 结果保存目录（workflow1专用）

# ==================== 主程序 ====================


def print_metrics_summary(metrics: dict):
    """打印指标摘要"""
    print("\n" + "=" * 70)
    print("资源使用统计")
    print("=" * 70)

    # 基本信息
    print(f"\n工作流名称: {metrics['workflow_name']}")
    print(f"进程ID: {metrics['pid']}")
    print(f"执行时长: {metrics['duration_seconds']} 秒")

    # CPU统计
    if 'cpu' in metrics['summary']:
        cpu = metrics['summary']['cpu']
        print(f"\n【CPU】")
        print(f"  进程CPU使用率: {cpu['process_cpu_percent']}% (平均), {cpu['peak_cpu_percent']}% (峰值)")
        print(f"  系统CPU使用率: {cpu['system_cpu_percent']}%")
        print(f"  用户态时间: {cpu['cpu_time_user']} 秒")
        print(f"  内核态时间: {cpu['cpu_time_system']} 秒")
        print(f"  CPU核心数: {cpu['cpu_count']}")

    # 内存统计
    if 'memory' in metrics['summary']:
        mem = metrics['summary']['memory']
        print(f"\n【内存】")
        print(f"  常驻内存(RSS): {mem['rss_mb']} MB (当前), {mem['peak_rss_mb']} MB (峰值)")
        print(f"  虚拟内存(VMS): {mem['vms_mb']} MB")
        print(f"  内存占用率: {mem['percent']}%")
        print(f"  系统总内存: {mem['system_total_gb']} GB")
        print(f"  系统可用内存: {mem['system_available_gb']} GB")

    # 存储IO统计
    if 'storage' in metrics['summary']:
        storage = metrics['summary']['storage']
        if storage['supported']:
            print(f"\n【存储IO】")
            print(f"  读取: {storage['read_mb']} MB ({storage['read_count']} 次)")
            print(f"  写入: {storage['write_mb']} MB ({storage['write_count']} 次)")
            print(f"  总IO: {storage['total_io_mb']} MB")
        else:
            print(f"\n【存储IO】不支持（系统限制）")

    # GPU统计
    if 'gpu' in metrics['summary']:
        gpu = metrics['summary']['gpu']
        if gpu['available']:
            print(f"\n【GPU】")
            print(f"  GPU数量: {gpu['gpu_count']}")
            print(f"  进程GPU显存: {gpu['process_gpu_memory_mb']} MB")
            if 'avg_utilization_percent' in gpu:
                print(f"  平均利用率: {gpu['avg_utilization_percent']}%")
            if 'gpus' in gpu:
                for gpu_info in gpu['gpus']:
                    print(f"  GPU {gpu_info['id']}: "
                          f"{gpu_info['utilization_percent']}% 利用率, "
                          f"{gpu_info['memory_used_mb']}/{gpu_info['memory_total_mb']} MB 显存, "
                          f"{gpu_info['temperature_c']}°C")
        else:
            print(f"\n【GPU】不可用")

    # 网络统计
    if 'network' in metrics['summary']:
        net = metrics['summary']['network']
        print(f"\n【网络】")
        print(f"  网络连接数: {net['connections_count']}")
        print(f"  发送: {net['sent_mb']} MB ({net['packets_sent']} 包)")
        print(f"  接收: {net['recv_mb']} MB ({net['packets_recv']} 包)")
        print(f"  总流量: {net['total_traffic_mb']} MB")
        print(f"  注: {net['note']}")

    # 节点统计
    if metrics['nodes']:
        print(f"\n【节点执行详情】")
        print(f"总节点数: {len(metrics['nodes'])}")
        for node in metrics['nodes']:
            print(f"\n  节点: {node['node_name']}")
            print(f"    执行时间: {node['duration_seconds']} 秒")
            if 'cpu' in node['metrics']:
                print(f"    CPU: {node['metrics']['cpu']['process_cpu_percent']}% (平均)")
            if 'memory' in node['metrics']:
                print(f"    内存: {node['metrics']['memory']['rss_mb']} MB")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    print("=" * 70)
    print("工作流执行器（带资源监控）")
    print("=" * 70)

    # 显示配置信息
    print("\n【监控配置】")
    enabled_metrics = [k.replace('enable_', '').upper() for k, v in MONITORING_CONFIG.items() if v]
    print(f"启用的监控指标: {', '.join(enabled_metrics)}")

    print(f"\n【工作流配置】")
    print(f"工作流: {WORKFLOW_NAME}")
    print(f"节点数量: {len(NODE_FUNCTIONS)}")
    print(f"用户输入: {USER_INPUT}")

    print("\n" + "-" * 70)
    print("开始执行工作流...")
    print("-" * 70 + "\n")

    # 使用 TracedWorkflowRunner 追踪节点执行
    with TracedWorkflowRunner(
        workflow_name=WORKFLOW_NAME,
        workflow_module_path=WORKFLOW_MODULE,
        node_functions=NODE_FUNCTIONS,
        **MONITORING_CONFIG
    ) as runner:
        try:
            # 执行工作流
            result = run_booking_workflow(USER_INPUT)

            # 输出工作流结果
            print("【工作流执行结果】")
            print(f"用户：{USER_INPUT}")
            print(f"\n助手：{result}")

        except Exception as e:
            print(f"\n工作流执行出错: {e}")
            import traceback
            traceback.print_exc()
            result = None

        # 获取追踪指标
        metrics = runner.get_metrics()

        # 打印指标摘要
        print_metrics_summary(metrics)

        # 保存结果到文件
        output_file = runner.save_results(output_dir=OUTPUT_DIR)
        print(f"\n详细结果已保存到: {output_file}")

    print("\n" + "=" * 70)
    print("执行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
