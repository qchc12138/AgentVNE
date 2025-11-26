#!/usr/bin/env python3
"""
SimuVNE 主程序入口
实现类似于simuGNN但有所不同的神经网络
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

from trainer import create_trainer
from model import create_model
from dataset import create_data_loaders


def set_seed(seed: int = 42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 展平嵌套的配置
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    # 处理设备配置
    if flat_config.get('device') == 'auto':
        flat_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return flat_config


def train_model(config_path: str, resume_from: str = None, **kwargs):
    """训练模型"""
    print("=" * 60)
    print("SimuVNE 神经网络训练")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    config.update(kwargs)  # 命令行参数覆盖配置文件
    
    print(f"配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 创建训练器
    trainer = create_trainer(**config)
    
    # 开始训练
    num_epochs = config.get('num_epochs', 100)
    trainer.train(num_epochs=num_epochs, resume_from=resume_from)
    
    print("\n训练完成！")


def test_model(config_path: str, checkpoint_path: str, **kwargs):
    """测试模型"""
    print("=" * 60)
    print("SimuVNE 神经网络测试")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    config.update(kwargs)
    
    # 检查检查点文件
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint.get('config', config)
    
    # 创建模型
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = create_model(
        input_dim=model_config.get('input_dim', 6),
        hidden_dim=model_config.get('hidden_dim', 64),
        hist_dim=model_config.get('hist_dim', 32)
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"从 {checkpoint_path} 加载模型")
    print(f"训练epoch: {checkpoint['epoch']}")
    print(f"最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")
    
    # 创建测试数据
    _, test_loader, _ = create_data_loaders(
        data_dir=config.get('data_dir', './data'),
        batch_size=1,  # 测试时使用批大小1
        train_ratio=0.8,
        hist_dim=config.get('hist_dim', 32),
        normalize_features=config.get('normalize_features', True)
    )
    
    # 测试模型
    total_samples = 0
    total_correct = 0
    
    print("\n开始测试...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # 只测试前10个样本
                break
                
            graph_i = batch['graphs_i'][0].to(device)
            graph_j = batch['graphs_j'][0].to(device)
            label = batch['labels'][0].to(device)
            
            # 前向传播（hist(S)在模型内部自动计算）
            output = model(graph_i, graph_j)
            
            # 计算预测
            pred = torch.sigmoid(output) > 0.5
            correct = (pred == label).float().sum().item()
            total_elements = label.numel()
            
            total_correct += correct
            total_samples += total_elements
            
            print(f"样本 {i+1}: 准确率 = {correct/total_elements:.4f}")
    
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
        print(f"\n总体测试准确率: {overall_accuracy:.4f}")
    
    print("测试完成！")


def create_sample_data(data_dir: str, num_samples: int = 100):
    """创建示例数据"""
    print(f"在 {data_dir} 创建 {num_samples} 个示例样本...")
    
    # 这会触发数据集的示例数据创建
    from dataset import GraphPairDataset
    dataset = GraphPairDataset(data_dir, hist_dim=32)
    
    print(f"示例数据创建完成，共 {len(dataset)} 个样本")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SimuVNE 神经网络训练和测试')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'create_data'], 
                       default='train', help='运行模式')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, 
                       help='恢复训练的检查点路径')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='测试用的检查点路径')
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='输出目录路径')
    parser.add_argument('--num_epochs', type=int, default=None, 
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=None, 
                       help='学习率')
    parser.add_argument('--device', type=str, default=None, 
                       help='设备 (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子')
    parser.add_argument('--num_samples', type=int, default=100, 
                       help='创建示例数据的样本数量')
    
    args = parser.parse_args()
    
    # 构建配置文件的绝对路径
    if not os.path.isabs(args.config):
        args.config = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    
    # 准备关键字参数
    kwargs = {}
    for key in ['data_dir', 'output_dir', 'num_epochs', 'batch_size', 'learning_rate', 'device']:
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    kwargs['seed'] = args.seed
    
    try:
        if args.mode == 'train':
            train_model(args.config, args.resume, **kwargs)
        elif args.mode == 'test':
            if args.checkpoint is None:
                print("测试模式需要指定 --checkpoint 参数")
                sys.exit(1)
            test_model(args.config, args.checkpoint, **kwargs)
        elif args.mode == 'create_data':
            data_dir = kwargs.get('data_dir', '/home/yc2/mrt/a/data')
            create_sample_data(data_dir, args.num_samples)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
