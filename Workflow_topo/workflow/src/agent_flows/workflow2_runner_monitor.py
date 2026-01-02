"""workflow2执行器（带资源监控）"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent_flows.workflows.workflow2 import run_camera_workflow
from src.agent_flows.observability.node_wrapper import TracedWorkflowRunner


# ==================== 配置区域 ====================

MONITORING_CONFIG = {
    "enable_cpu": True,
    "enable_memory": True,
    "enable_storage": True,
    "enable_gpu": True,
    "enable_network": True,
}

WORKFLOW_NAME = "workflow2"
WORKFLOW_MODULE = "src.agent_flows.workflows.workflow2"
USER_INPUT = "请分析路口摄像头画面，判断车流是否拥堵，若有噪声请做轻量去噪"

NODE_FUNCTIONS = {
    "节点1-意图理解": "intent_understanding_node",
    "节点2-采集": "camera_capture_node",
    "节点3-预处理": "preprocessing_node",
    "节点4-特征": "feature_extraction_node",
    "节点5-分析": "task_execution_node",
    "节点6-总结": "summarization_node",
}

OUTPUT_DIR = "results/workflow2"

# ==================== 主程序 ====================


def print_metrics_summary(metrics: dict):
    print("\n" + "=" * 70)
    print("资源使用统计")
    print("=" * 70)

    print(f"\n工作流名称: {metrics['workflow_name']}")
    print(f"进程ID: {metrics['pid']}")
    print(f"执行时长: {metrics['duration_seconds']} 秒")

    if "cpu" in metrics.get("summary", {}):
        cpu = metrics["summary"]["cpu"]
        print("\n【CPU】")
        print(f"  进程CPU使用率: {cpu['process_cpu_percent']}% (平均), {cpu['peak_cpu_percent']}% (峰值)")
        print(f"  系统CPU使用率: {cpu['system_cpu_percent']}%")
        print(f"  用户态时间: {cpu['cpu_time_user']} 秒")
        print(f"  内核态时间: {cpu['cpu_time_system']} 秒")
        print(f"  CPU核心数: {cpu['cpu_count']}")

    if "memory" in metrics.get("summary", {}):
        mem = metrics["summary"]["memory"]
        print("\n【内存】")
        print(f"  常驻内存(RSS): {mem['rss_mb']} MB (当前), {mem['peak_rss_mb']} MB (峰值)")
        print(f"  虚拟内存(VMS): {mem['vms_mb']} MB")
        print(f"  内存占用率: {mem['percent']}%")
        print(f"  系统总内存: {mem['system_total_gb']} GB")
        print(f"  系统可用内存: {mem['system_available_gb']} GB")

    if "storage" in metrics.get("summary", {}):
        storage = metrics["summary"]["storage"]
        if storage.get("supported"):
            print("\n【存储IO】")
            print(f"  读取: {storage['read_mb']} MB ({storage['read_count']} 次)")
            print(f"  写入: {storage['write_mb']} MB ({storage['write_count']} 次)")
            print(f"  总IO: {storage['total_io_mb']} MB")
        else:
            print("\n【存储IO】不支持（系统限制）")

    if "gpu" in metrics.get("summary", {}):
        gpu = metrics["summary"]["gpu"]
        if gpu.get("available"):
            print("\n【GPU】")
            print(f"  GPU数量: {gpu['gpu_count']}")
            print(f"  进程GPU显存: {gpu['process_gpu_memory_mb']} MB")
            if "avg_utilization_percent" in gpu:
                print(f"  平均利用率: {gpu['avg_utilization_percent']}%")
            if "gpus" in gpu:
                for g in gpu["gpus"]:
                    print(
                        f"  GPU {g['id']}: {g['utilization_percent']}% 利用率, "
                        f"{g['memory_used_mb']}/{g['memory_total_mb']} MB 显存, "
                        f"{g['temperature_c']}°C"
                    )
        else:
            print("\n【GPU】不可用")

    if "network" in metrics.get("summary", {}):
        net = metrics["summary"]["network"]
        print("\n【网络】")
        print(f"  网络连接数: {net['connections_count']}")
        print(f"  发送: {net['sent_mb']} MB ({net['packets_sent']} 包)")
        print(f"  接收: {net['recv_mb']} MB ({net['packets_recv']} 包)")
        print(f"  总流量: {net['total_traffic_mb']} MB")
        print(f"  注: {net['note']}")

    if metrics.get("nodes"):
        print("\n【节点执行详情】")
        print(f"总节点数: {len(metrics['nodes'])}")
        for node in metrics["nodes"]:
            print(f"\n  节点: {node['node_name']}")
            print(f"    执行时间: {node['duration_seconds']} 秒")
            if "cpu" in node.get("metrics", {}):
                print(f"    CPU: {node['metrics']['cpu']['process_cpu_percent']}% (平均)")
            if "memory" in node.get("metrics", {}):
                print(f"    内存: {node['metrics']['memory']['rss_mb']} MB")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("workflow2 执行器（带资源监控）")
    print("=" * 70)

    print("\n【监控配置】")
    enabled = [k.replace("enable_", "").upper() for k, v in MONITORING_CONFIG.items() if v]
    print(f"启用的监控指标: {', '.join(enabled)}")

    print("\n【工作流配置】")
    print(f"工作流: {WORKFLOW_NAME}")
    print(f"节点数量: {len(NODE_FUNCTIONS)}")
    print(f"用户输入: {USER_INPUT}")

    print("\n" + "-" * 70)
    print("开始执行工作流...")
    print("-" * 70 + "\n")

    with TracedWorkflowRunner(
        workflow_name=WORKFLOW_NAME,
        workflow_module_path=WORKFLOW_MODULE,
        node_functions=NODE_FUNCTIONS,
        **MONITORING_CONFIG,
    ) as runner:
        try:
            result = run_camera_workflow(USER_INPUT)
            print("【工作流执行结果】")
            print(f"用户：{USER_INPUT}")
            print(f"\n助手：{result}")
        except Exception as e:
            print(f"\n工作流执行出错: {e}")
            import traceback

            traceback.print_exc()
        metrics = runner.get_metrics()
        print_metrics_summary(metrics)
        output_file = runner.save_results(output_dir=OUTPUT_DIR)
        print(f"\n详细结果已保存到: {output_file}")

    print("\n" + "=" * 70)
    print("执行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
