"""workflow2运行器：测试摄像头图像处理工作流"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent_flows.workflows.workflow2 import run_camera_workflow, run_camera_workflow_with_details


def main():
    """运行workflow2的主函数"""
    default_user_input = "查看路口摄像头，估计当前车流是否拥堵，画面如果有噪声先做降噪"

    print(f"用户：{default_user_input}\n")

    result = run_camera_workflow(default_user_input)
    print(f"助手：{result}\n")


if __name__ == "__main__":
    main()
