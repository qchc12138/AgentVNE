#!/usr/bin/env python
"""节点优化器运行脚本"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.node_optimizer_agent import run_node_optimizer, save_results

if __name__ == "__main__":
    # 运行分析
    output = run_node_optimizer()

    # 输出总结
    print(output.summary)

    # 保存详细结果
    output_path = save_results(output)
    print(f"\n详细分析结果已保存到: {output_path}")
