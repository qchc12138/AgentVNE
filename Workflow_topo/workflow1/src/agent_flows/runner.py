"""工作流运行器：用于测试和运行工作流"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent_flows.workflows.workflow1 import run_booking_workflow, run_booking_workflow_with_details


def main():
    """运行工作流的主函数"""
    # 默认用户输入
    default_user_input = "请帮我在和平路搜索酒店，预算在200到300元之间"

    print(f"用户：{default_user_input}")
    print()

    # 调用工作流（只获取最终回复）
    result = run_booking_workflow(default_user_input)

    # 只输出最终的助手回复
    print(f"助手：{result}")
    print()


if __name__ == "__main__":
    main()
