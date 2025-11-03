#!/usr/bin/env python3
"""
测试脚本：加载fine_tuning模型，生成任务并放置
打印：概率矩阵、采样动作、放置前后SN剩余容量
"""
import torch
import numpy as np
from model import SimuVNE
from fine_tuning import ValueNet, PPOAgent
from env import SimuVNEEnv, WorkflowGenerator


def print_probability_matrix(probs_matrix, vn_size, sn_size):
    """打印概率矩阵（格式化显示）"""
    print("  【概率矩阵】 (VN节点 × SN节点)")
    print("  " + "".join([f"SN{i:2d}  " for i in range(min(sn_size, 10))]))
    for i in range(vn_size):
        row_str = f"  VN{i}: "
        for j in range(min(sn_size, 10)):
            row_str += f"{probs_matrix[i, j].item():.3f} "
        if sn_size > 10:
            row_str += "..."
        print(row_str)
    if vn_size > 10:
        print("  ...")


def print_sn_residual(env, title="SN剩余容量"):
    """打印SN节点的剩余资源"""
    print(f"  【{title}】")
    for n in sorted(env.G_sn.nodes)[:10]:  # 只显示前10个节点
        nd = env.G_sn.nodes[n]
        print(f"    节点{n}: CPU={nd['cpu_res']:.2f}, Mem={nd['mem_res']:.2f}, Disk={nd['disk_res']:.2f}")
    if len(env.G_sn.nodes) > 10:
        print(f"    ... (共 {len(env.G_sn.nodes)} 个节点)")


def test_placement_with_model(model_path=None, num_tasks=5):
    """
    测试任务放置流程
    
    Args:
        model_path: 模型checkpoint路径，None则使用随机初始化
        num_tasks: 测试任务数量
    """
    print("=" * 80)
    print("测试：加载模型并进行任务放置")
    print("=" * 80)
    
    # 配置
    sn_path = '/home/zrz/SimuVNE/topo/SN_topology.json'
    workflow_types = {
        'workflow1': '/home/zrz/SimuVNE/workflow_topo/workflow1_topo.json',
    }
    device = 'cpu'
    
    # 创建环境
    print(f"\n【1. 初始化环境】")
    env = SimuVNEEnv(
        sn_topology_path=sn_path,
        device=device,
        penalty=-150.0,
        max_arrived_tasks=num_tasks
    )
    env.reset()
    sn_capacity = env.get_sn_max_capacity()
    
    print(f"  SN拓扑: {len(env.G_sn.nodes)} 个节点, {len(env.G_sn.edges)} 条边")
    print(f"  SN最大容量: CPU={sn_capacity['cpu_max']}, Mem={sn_capacity['mem_max']}, Disk={sn_capacity['disk_max']}")
    
    # 创建工作流生成器
    wf_gen = WorkflowGenerator(
        workflow_types=workflow_types,
        arrival_rate=1.0,  # 每个时间单位必定到达
        mean_lifetime=10.0,
        seed=42,
        sn_capacity_for_norm=sn_capacity
    )
    
    # 加载或创建模型
    print(f"\n【2. 加载模型】")
    policy = SimuVNE()
    value_net = ValueNet()
    
    if model_path:
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            policy.load_state_dict(state_dict, strict=False)
            print(f"  ✓ 已加载模型: {model_path}")
        except Exception as e:
            print(f"  ✗ 加载模型失败: {e}")
            print(f"  → 使用随机初始化的模型")
    else:
        print(f"  → 使用随机初始化的模型")
    
    agent = PPOAgent(policy, value_net, device=device)
    policy.eval()  # 设置为评估模式
    
    # 打印初始SN状态
    print(f"\n【3. 初始SN状态】")
    print_sn_residual(env, "SN初始剩余容量")
    
    # 开始放置任务
    print(f"\n{'=' * 80}")
    print(f"【4. 开始任务放置】 (共 {num_tasks} 个任务)")
    print(f"{'=' * 80}")
    
    placed_count = 0
    failed_count = 0
    
    for task_id in range(num_tasks):
        print(f"\n{'-' * 80}")
        print(f"任务 #{task_id}")
        print(f"{'-' * 80}")
        
        # 生成VN任务
        wf_type = wf_gen.sample_workflow_type()
        vn = wf_gen.load_workflow_graph(wf_type)
        lifetime = wf_gen.sample_lifetime()
        
        print(f"  任务类型: {wf_type}")
        print(f"  VN节点数: {vn.x.size(0)}")
        print(f"  生存时间: {lifetime:.1f}")
        
        # 打印VN需求（归一化）
        print(f"\n  【VN节点需求】 (归一化)")
        for i in range(min(vn.x.size(0), 6)):
            feats = vn.x[i]
            print(f"    节点{i}: cpu={feats[0].item():.3f}, mem={feats[1].item():.3f}, disk={feats[2].item():.3f}")
        
        # 获取当前SN状态
        sn_state = env.get_sn_state()
        
        # 策略网络推理
        print(f"\n  【策略网络推理】")
        with torch.no_grad():
            vn_input = vn.to(device)
            sn_input = sn_state.to(device)
            
            # 获取概率矩阵
            probs_matrix = policy(vn_input, sn_input)
            print(f"  输出形状: {probs_matrix.shape} (VN节点 × SN节点)")
        
        # 打印概率矩阵
        print_probability_matrix(probs_matrix, vn.x.size(0), sn_state.x.size(0))
        
        # 采样放置动作
        print(f"\n  【采样放置动作】")
        mapping, logprob, value = agent.act(vn, sn_state)
        print(f"  映射: {mapping}")
        print(f"  Log概率: {logprob.item():.4f}")
        print(f"  状态价值: {value.item():.4f}")
        
        # 打印放置前SN状态
        print(f"\n  【放置前SN状态】")
        print_sn_residual(env, "放置前SN剩余容量")
        
        # 尝试放置（带重试）
        print(f"\n  【尝试放置】")
        max_retries = 3
        success = False
        r_t = None
        
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                # 重新采样
                mapping, logprob, value = agent.act(vn, sn_state)
                print(f"  重试 {attempt-1}: 新映射 {mapping}")
            
            success, r_t = env.try_place_task(vn, mapping, lifetime, task_id)
            
            if success:
                print(f"  ✓ 放置成功 (尝试 {attempt} 次)")
                print(f"    r_t = {r_t:.3f}")
                placed_count += 1
                break
        
        if not success:
            print(f"  ✗ 放置失败 (尝试 {max_retries} 次)")
            print(f"    penalty = {r_t:.3f}")
            failed_count += 1
        
        # 打印放置后SN状态
        print(f"\n  【放置后SN状态】")
        print_sn_residual(env, "放置后SN剩余容量")
        print(f"  当前存活任务数: {len(env.active_workflows)}")
        
        # 推进时间（模拟任务过期）
        env.arrived_count += 1
        env.step_time(time_delta=2.0)  # 推进2个时间单位
    
    # 统计结果
    print(f"\n{'=' * 80}")
    print(f"【5. 测试结果统计】")
    print(f"{'=' * 80}")
    print(f"  总任务数: {num_tasks}")
    print(f"  成功放置: {placed_count} ({placed_count/num_tasks*100:.1f}%)")
    print(f"  放置失败: {failed_count} ({failed_count/num_tasks*100:.1f}%)")
    print(f"  最终存活任务数: {len(env.active_workflows)}")
    
    # 打印最终SN状态
    print(f"\n【6. 最终SN状态】")
    print_sn_residual(env, "最终SN剩余容量")
    
    print(f"\n{'=' * 80}")
    print("测试完成！")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    # 测试场景1: 使用随机初始化模型
    print("\n【场景1: 随机初始化模型】")
    test_placement_with_model(model_path=None, num_tasks=3)
    
    # 测试场景2: 使用预训练模型（如果存在）
    # pretrained_model = '/home/zrz/SimuVNE/pretrain_outputs/checkpoint_best.pt'
    # print("\n\n【场景2: 预训练模型】")
    # test_placement_with_model(model_path=pretrained_model, num_tasks=3)
    
    # 测试场景3: 使用fine-tuning模型（如果存在）
    # 查找最新的fine-tuning输出
    import os
    import glob
    finetuning_dir = '/home/zrz/SimuVNE/finetuning_putput'
    runs = sorted(glob.glob(f"{finetuning_dir}/run_*/policy_network.pth"))
    if runs:
        latest_model = runs[-1]
        print(f"\n\n【场景3: Fine-tuning模型】")
        print(f"最新模型: {latest_model}")
        test_placement_with_model(model_path=latest_model, num_tasks=3)
    else:
        print(f"\n未找到fine-tuning模型，跳过场景3")

