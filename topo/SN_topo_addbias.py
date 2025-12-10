#!/usr/bin/env python3
"""
为 SN 拓扑添加偏置字段 bias_cpu / bias_bandwidth。
规则：
- max_cpu = 所有节点 cpu 最大值
- 若 node.id == constraint_node: bias_cpu = bias * max_cpu
- 否则 bias_cpu = 0
- bias_bandwidth = comm_bandwidth
默认读写同目录下 SN_topology.json -> SN_topology_2.json
"""

import argparse
import json
import os
from typing import Dict, List


def add_bias_to_topology(
    input_path: str,
    output_path: str,
    bias: float = 0.4,
    constraint_node: int = 6
) -> None:
    """读取 input_path，写入带偏置字段的拓扑到 output_path。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        topo = json.load(f)

    nodes: List[Dict] = topo.get('nodes', [])
    if not nodes:
        raise ValueError("拓扑文件中没有 nodes 字段或为空")

    max_cpu = max(float(n.get('cpu', 0.0)) for n in nodes)

    for n in nodes:
        node_id = int(n.get('id', -1))
        comm_bw = float(n.get('comm_bandwidth', n.get('bandwidth', 0.0)))
        if node_id == constraint_node:
            bias_cpu = bias * max_cpu
        else:
            bias_cpu = 0.0
        n['bias_cpu'] = bias_cpu
        n['bias_bandwidth'] = comm_bw

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topo, f, indent=2, ensure_ascii=False)

    print(f"生成完成 -> {output_path} (bias={bias}, constraint_node={constraint_node})")


def main() -> int:
    parser = argparse.ArgumentParser(description="为 SN 拓扑添加偏置字段")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, 'SN_topology.json')
    default_output = os.path.join(script_dir, 'SN_topology_2.json')

    parser.add_argument('--bias', type=float, default=0.4, help='偏置系数，默认 0.4')
    parser.add_argument('--constraint_node', type=int, default=6, help='约束节点 ID，默认 6')
    parser.add_argument('--input', type=str, default=default_input, help='输入拓扑路径')
    parser.add_argument('--output', type=str, default=default_output, help='输出拓扑路径')

    args = parser.parse_args()

    add_bias_to_topology(
        input_path=args.input,
        output_path=args.output,
        bias=args.bias,
        constraint_node=args.constraint_node
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

