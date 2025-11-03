#!/usr/bin/env python3
"""
测试归一化效果的脚本
"""
import sys
import torch
from env import SimuVNEEnv, WorkflowGenerator

def test_normalization():
    print("="*60)
    print("测试特征归一化")
    print("="*60)
    
    # 1. 创建环境
    sn_path = '/home/zrz/SimuVNE/topo/SN_topology.json'
    workflow_types = {
        'workflow1': '/home/zrz/SimuVNE/workflow_topo/workflow1_topo.json',
    }
    
    env = SimuVNEEnv(
        sn_topology_path=sn_path,
        device='cpu',
        penalty=-50.0,
        max_arrived_tasks=20
    )
    env.reset()
    
    # 2. 获取SN最大容量
    sn_capacity = env.get_sn_max_capacity()
    print("\n【SN最大容量】")
    for key, val in sn_capacity.items():
        print(f"  {key}: {val:.2f}")
    
    # 3. 测试SN状态归一化
    print("\n【SN状态归一化检查】")
    sn_state = env.get_sn_state()
    print(f"  SN节点数: {sn_state.x.size(0)}")
    print(f"  SN特征维度: {sn_state.x.size(1)}")
    print(f"  SN特征统计:")
    print(f"    CPU (dim 0): min={sn_state.x[:, 0].min():.4f}, max={sn_state.x[:, 0].max():.4f}, mean={sn_state.x[:, 0].mean():.4f}")
    print(f"    Memory (dim 1): min={sn_state.x[:, 1].min():.4f}, max={sn_state.x[:, 1].max():.4f}, mean={sn_state.x[:, 1].mean():.4f}")
    print(f"    Disk (dim 2): min={sn_state.x[:, 2].min():.4f}, max={sn_state.x[:, 2].max():.4f}, mean={sn_state.x[:, 2].mean():.4f}")
    print(f"    Bandwidth (dim 3): min={sn_state.x[:, 3].min():.4f}, max={sn_state.x[:, 3].max():.4f}, mean={sn_state.x[:, 3].mean():.4f}")
    print(f"    Comm BW (dim 4): min={sn_state.x[:, 4].min():.4f}, max={sn_state.x[:, 4].max():.4f}, mean={sn_state.x[:, 4].mean():.4f}")
    
    # 4. 测试VN特征归一化
    print("\n【VN特征归一化检查】")
    wf_gen = WorkflowGenerator(
        workflow_types=workflow_types,
        arrival_rate=0.05,
        mean_lifetime=10.0,
        seed=42,
        sn_capacity_for_norm=sn_capacity
    )
    
    vn = wf_gen.load_workflow_graph('workflow1')
    print(f"  VN节点数: {vn.x.size(0)}")
    print(f"  VN特征维度: {vn.x.size(1)}")
    print(f"  VN特征统计:")
    print(f"    CPU (dim 0): min={vn.x[:, 0].min():.4f}, max={vn.x[:, 0].max():.4f}, mean={vn.x[:, 0].mean():.4f}")
    print(f"    Memory (dim 1): min={vn.x[:, 1].min():.4f}, max={vn.x[:, 1].max():.4f}, mean={vn.x[:, 1].mean():.4f}")
    print(f"    Disk (dim 2): min={vn.x[:, 2].min():.4f}, max={vn.x[:, 2].max():.4f}, mean={vn.x[:, 2].mean():.4f}")
    print(f"    Bandwidth (dim 3): min={vn.x[:, 3].min():.4f}, max={vn.x[:, 3].max():.4f}, mean={vn.x[:, 3].mean():.4f}")
    print(f"    Comm BW (dim 4): min={vn.x[:, 4].min():.4f}, max={vn.x[:, 4].max():.4f}, mean={vn.x[:, 4].mean():.4f}")
    
    # 5. 打印实际数值（前3个节点）
    print("\n【SN前3个节点的归一化特征】")
    for i in range(min(3, sn_state.x.size(0))):
        print(f"  节点 {i}: {sn_state.x[i].tolist()}")
    
    print("\n【VN前3个节点的归一化特征】")
    for i in range(min(3, vn.x.size(0))):
        print(f"  节点 {i}: {vn.x[i].tolist()}")
    
    # 6. 检查归一化后的值是否在合理范围
    print("\n【归一化范围检查】")
    sn_max = sn_state.x.max().item()
    sn_min = sn_state.x.min().item()
    vn_max = vn.x.max().item()
    vn_min = vn.x.min().item()
    
    print(f"  SN特征范围: [{sn_min:.4f}, {sn_max:.4f}]")
    print(f"  VN特征范围: [{vn_min:.4f}, {vn_max:.4f}]")
    
    if sn_max <= 1.1 and sn_min >= -0.1:
        print("  ✓ SN特征归一化正常（大部分在[0,1]范围内）")
    else:
        print(f"  ✗ SN特征归一化异常！超出预期范围")
    
    if vn_max <= 1.5 and vn_min >= -0.1:
        print("  ✓ VN特征归一化正常（大部分在合理范围内）")
    else:
        print(f"  ✗ VN特征归一化异常！超出预期范围")
    
    # 7. 测试资源消耗后的归一化
    print("\n【测试资源消耗后的归一化】")
    # 模拟放置一个任务
    mapping = {i: i % sn_state.x.size(0) for i in range(vn.x.size(0))}
    lifetime = 10.0
    success, r_t = env.try_place_task(vn, mapping, lifetime, task_id=0)
    
    if success:
        print(f"  任务放置成功，r_t={r_t:.3f}")
        sn_state_after = env.get_sn_state()
        print(f"  资源消耗后SN特征统计:")
        print(f"    CPU: min={sn_state_after.x[:, 0].min():.4f}, max={sn_state_after.x[:, 0].max():.4f}, mean={sn_state_after.x[:, 0].mean():.4f}")
        print(f"    Memory: min={sn_state_after.x[:, 1].min():.4f}, max={sn_state_after.x[:, 1].max():.4f}, mean={sn_state_after.x[:, 1].mean():.4f}")
        print(f"    Disk: min={sn_state_after.x[:, 2].min():.4f}, max={sn_state_after.x[:, 2].max():.4f}, mean={sn_state_after.x[:, 2].mean():.4f}")
        
        # 检查资源是否减少
        cpu_decreased = (sn_state_after.x[:, 0] < sn_state.x[:, 0]).any()
        if cpu_decreased:
            print("  ✓ CPU资源正确扣减")
        else:
            print("  ✗ CPU资源未扣减（可能是映射问题）")
    else:
        print(f"  任务放置失败，r_t={r_t:.3f}")
    
    print("\n" + "="*60)
    print("归一化测试完成！")
    print("="*60)

if __name__ == '__main__':
    test_normalization()

