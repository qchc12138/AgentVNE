#!/usr/bin/env python3
"""
预训练脚本
使用预先生成的数据集对 SimuVNE 模型进行 BCEWithLogitsLoss 预训练。

使用前请先运行 dataset_generate.py 生成预训练数据集。

同时验证给模型的图输入（节点特征与 edge_index）是否正确。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard不可用，将跳过TensorBoard日志记录")

from tqdm import tqdm
from model_2 import SimuVNE
from torch_geometric.data import Data


class PretrainTrainer:
    """预训练器"""
    
    def __init__(self, model, train_loader, val_loader=None, 
                 learning_rate=0.001, weight_decay=1e-5,
                 device='cuda', output_dir='./pretrain_outputs',
                 use_tensorboard=True):
        """
        初始化预训练器
        
        Args:
            model: SimuVNE模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 设备
            output_dir: 输出目录
            use_tensorboard: 是否使用TensorBoard
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 损失函数：KL散度 + MSE
        def kl_mse_loss(output, label):
            eps = 1e-8
            # KL(Q||P) 按行求和后取平均（output/label 为每行概率分布）
            p = torch.clamp(output, min=eps)
            q = torch.clamp(label,  min=eps)
            kl_row = torch.sum(q * (torch.log(q) - torch.log(p)), dim=1)
            kl = kl_row.mean()
            # MSE 元素级平均
            mse = torch.mean((output - label) ** 2)
            # return kl + mse*100
            # return kl
            return mse
        self.criterion = kl_mse_loss
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.5
        )
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            log_dir = os.path.join(output_dir, 'logs')
            self.writer = SummaryWriter(log_dir)
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("警告: TensorBoard不可用，日志将不会被记录到TensorBoard")
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"\n预训练器初始化完成:")
        print(f"  设备: {device}")
        print(f"  学习率: {learning_rate}")
        print(f"  权重衰减: {weight_decay}")
        print(f"  输出目录: {output_dir}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train {epoch+1}", leave=False)):
            batch_loss = 0
            batch_size = len(batch['workflow_graphs'])
            
            # 逐个样本处理（因为图大小不同）
            for i in range(batch_size):
                workflow_graph = batch['workflow_graphs'][i].to(self.device)
                substrate_graph = batch['substrate_graphs'][i].to(self.device)
                label = batch['labels'][i].to(self.device)  # (N1, N2)
                
                # 前向传播
                output = self.model(workflow_graph, substrate_graph)  # (N1, N2)
                
                # 计算损失
                loss = self.criterion(output, label)
                batch_loss += loss
            
            # 平均损失
            batch_loss = batch_loss / batch_size
            
            # 反向传播
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            # 打印进度
            # if (batch_idx + 1) % 10 == 0:
            #     tqdm.write(f"  Train batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {batch_loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """若无验证集，直接返回 None。"""
        if not self.val_loader:
            return None
        self.model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Valid {epoch+1}", leave=False):
                batch_loss = 0
                batch_size = len(batch['workflow_graphs'])
                for i in range(batch_size):
                    workflow_graph = batch['workflow_graphs'][i].to(self.device)
                    substrate_graph = batch['substrate_graphs'][i].to(self.device)
                    label = batch['labels'][i].to(self.device)
                    output = self.model(workflow_graph, substrate_graph)
                    loss = self.criterion(output, label)
                    batch_loss += loss
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
                num_batches += 1
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # 保存最新模型
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最优模型，额外保存
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"  保存最优模型: {best_path}")
        
        # 定期保存epoch检查点
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, epoch_path)
    
    def train(self, num_epochs):
        """
        执行预训练
        
        Args:
            num_epochs: 训练轮数
        """
        print(f"\n{'='*60}")
        print(f"开始预训练 - 共 {num_epochs} 个epoch")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print("-" * 60)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证（可选）
            val_loss = self.validate(epoch)
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time
            
            # 打印统计信息（仅训练损失）
            print(f"\nEpoch [{epoch + 1}/{num_epochs}] 完成:")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  耗时: {epoch_time:.2f}秒")
            
            # TensorBoard记录
            if self.use_tensorboard:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 保存检查点
            is_best = False
            if val_loss is not None:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"  *** 新的最优验证损失: {val_loss:.6f} ***")
            self.save_checkpoint(epoch, val_loss if val_loss is not None else train_loss, is_best)
        
        # 训练完成
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"预训练完成！")
        print(f"{'='*60}")
        print(f"总耗时: {total_time/60:.2f}分钟")
        print(f"最终训练损失: {self.train_losses[-1]:.6f}")
        print(f"模型保存在: {self.output_dir}")
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'num_epochs': num_epochs
        }
        
        history_file = os.path.join(self.output_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"训练历史保存在: {history_file}")
        
        if self.use_tensorboard:
            self.writer.close()


def load_pretrain_dataset(dataset_path: str) -> Tuple[List[Dict], Dict]:
    """从文件加载预训练数据集
    
    Args:
        dataset_path: 数据集文件路径（.pt 格式）
    
    Returns:
        samples: 样本列表，每个样本包含 'workflow_graph', 'substrate_graph', 'label'
        info: 数据集信息字典
    """
    print(f"从 {dataset_path} 加载数据集...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}\n请先运行 dataset_generate.py 生成数据集")
    
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    samples = data['samples']
    info = data.get('info', {})
    
    print(f"  加载了 {len(samples)} 个样本")
    if info:
        print(f"  数据集信息:")
        print(f"    样本数: {info.get('num_samples', 'N/A')}")
        print(f"    是否归一化: {info.get('normalized', False)}")
        if 'sn_max_capacity' in info:
            cap = info['sn_max_capacity']
            print(f"    归一化参数 (SN最大容量):")
            print(f"      CPU: {cap.get('cpu_max', 'N/A')}")
            print(f"      Memory: {cap.get('mem_max', 'N/A')}")
            print(f"      Disk: {cap.get('disk_max', 'N/A')}")
            print(f"      Bandwidth: {cap.get('bw_max', 'N/A')}")
            print(f"      Comm Bandwidth: {cap.get('comm_bw_max', 'N/A')}")
    
    return samples, info


def _collate_samples(batch: List[Dict]) -> Dict[str, List]:
    """变长图的简单 collate：返回列表，训练时逐样本前向。"""
    return {
        'workflow_graphs': [item['workflow_graph'] for item in batch],
        'substrate_graphs': [item['substrate_graph'] for item in batch],
        'labels': [item['label'] for item in batch],
    }


def create_pretrain_dataloader(samples: List[Dict], batch_size: int = 16):
    """创建仅训练用的 DataLoader（不划分验证集）。"""
    from torch.utils.data import DataLoader
    train_loader = DataLoader(samples, batch_size=batch_size, shuffle=True, collate_fn=_collate_samples)
    return train_loader


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SimuVNE 预训练脚本')
    parser.add_argument('--dataset', type=str,
                       default='/home/zrz/SimuVNE/pretrain_data/pretrain_dataset.pt',
                       help='预训练数据集文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='/home/zrz/SimuVNE/pretrain_outputs',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='批大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--test_dataset', type=str,
                       default='/home/zrz/SimuVNE/pretrain_data/pretrain_dataset.pt',
                       help='测试样本文件路径（单条）')
    parser.add_argument('--input_dim', type=int, default=6,
                       help='输入特征维度')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='隐藏层维度')
    parser.add_argument('--hist_dim', type=int, default=32,
                       help='直方图维度')
    parser.add_argument('--num_nodes_j', type=int, default=None,
                       help='SN节点数量（如果为None，将从数据集中自动检测）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu)，默认自动选择')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'dataset_path': args.dataset,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'test_dataset_path': args.test_dataset,
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'hist_dim': args.hist_dim,
        'device': args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'),
    }

    print("\n预训练配置:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # 加载预训练数据集
    print("\n" + "="*60)
    print("加载预训练数据集...")
    print("="*60)
    samples, dataset_info = load_pretrain_dataset(config['dataset_path'])
    print(f"总样本数: {len(samples)}")

    # 检测或设置 num_nodes_j
    if args.num_nodes_j is None:
        # 从数据集中自动检测最大 SN 节点数
        max_sn_nodes = 0
        for sample in samples:
            sn_nodes = sample['substrate_graph'].x.size(0)
            if sn_nodes > max_sn_nodes:
                max_sn_nodes = sn_nodes
        num_nodes_j = max_sn_nodes
        print(f"自动检测到最大 SN 节点数: {num_nodes_j}")
    else:
        num_nodes_j = args.num_nodes_j
        print(f"使用指定的 SN 节点数: {num_nodes_j}")
    
    config['num_nodes_j'] = num_nodes_j

    # 创建数据加载器（全量作为训练集）
    print("\n" + "="*60)
    print("创建训练数据加载器...")
    print("="*60)
    train_loader = create_pretrain_dataloader(samples=samples, batch_size=config['batch_size'])

    # 创建模型
    print("\n" + "="*60)
    print("创建模型...")
    print("="*60)
    model = SimuVNE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        hist_dim=config['hist_dim'],
        num_nodes_j=config['num_nodes_j']
    )

    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {num_params:,}")

    # 创建训练器
    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device'],
        output_dir=config['output_dir'],
        use_tensorboard=True
    )

    # 开始训练（BCEWithLogitsLoss 损失）
    trainer.train(num_epochs=config['num_epochs'])

    print("\n预训练任务完成！")

    # 加载测试样本并进行前向传播
    print("\n" + "="*60)
    print("加载测试样本并进行前向传播...")
    print("="*60)
    test_samples, test_info = load_pretrain_dataset(config['test_dataset_path'])
    if len(test_samples) == 0:
        print("测试样本为空，跳过测试。")
        return
    
    # 打印测试数据集信息
    print(f"\n测试数据集信息:")
    if test_info:
        print(f"  数据类型: {test_info.get('type', 'N/A')}")
        print(f"  样本数: {len(test_samples)}")
        if 'sn_max_capacity' in test_info:
            print(f"  归一化参数 (SN最大容量):")
            cap = test_info['sn_max_capacity']
            print(f"    CPU: {cap.get('cpu_max', 'N/A')}")
            print(f"    Memory: {cap.get('mem_max', 'N/A')}")
            print(f"    Disk: {cap.get('disk_max', 'N/A')}")
            print(f"    Bandwidth: {cap.get('bw_max', 'N/A')}")
            print(f"    Comm Bandwidth: {cap.get('comm_bw_max', 'N/A')}")
    
    test_sample = test_samples[0]
    workflow_graph = test_sample['workflow_graph'].to(config['device'])
    substrate_graph = test_sample['substrate_graph'].to(config['device'])
    label = test_sample['label'].to(config['device'])
    
    # 打印图结构信息
    print(f"\n图结构信息:")
    print(f"  Workflow 图:")
    print(f"    节点数: {workflow_graph.x.size(0)}")
    print(f"    边数: {workflow_graph.edge_index.size(1)}")
    print(f"    节点特征维度: {workflow_graph.x.size(1)}")
    print(f"    节点特征形状: {workflow_graph.x.shape}")
    print(f"  Substrate 图:")
    print(f"    节点数: {substrate_graph.x.size(0)}")
    print(f"    边数: {substrate_graph.edge_index.size(1)}")
    print(f"    节点特征维度: {substrate_graph.x.size(1)}")
    print(f"    节点特征形状: {substrate_graph.x.shape}")
    print(f"  标签:")
    print(f"    形状: {label.shape}")
    print(f"    总和: {label.sum().item():.6f}")
    
    # 打印节点特征详情（前3个节点）
    print(f"\nWorkflow 节点特征 (前3个节点):")
    print(f"  特征顺序: [cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]")
    for i in range(min(3, workflow_graph.x.size(0))):
        feat = workflow_graph.x[i].cpu().numpy()
        print(f"  节点 {i}: {feat}")
    
    print(f"\nSubstrate 节点特征 (前5个节点):")
    print(f"  特征顺序: [cpu, memory, disk, bandwidth, comm_bandwidth, 0.0]")
    for i in range(min(5, substrate_graph.x.size(0))):
        feat = substrate_graph.x[i].cpu().numpy()
        print(f"  节点 {i}: {feat}")
    
    # 打印标签详情（前3行）
    print(f"\n标签矩阵 (前3行，每行是一个 VN 节点对所有 SN 节点的分配概率):")
    label_np = label.cpu().numpy()
    for i in range(min(3, label.size(0))):
        print(f"  VN节点 {i}: {label_np[i]}")
    
    # 打印标签的统计信息
    print(f"\n标签统计信息:")
    print(f"  最小值: {label.min().item():.6f}")
    print(f"  最大值: {label.max().item():.6f}")
    print(f"  均值: {label.mean().item():.6f}")
    print(f"  每行总和 (应该相同): {label.sum(dim=1).cpu().numpy()}")
    
    # 找出概率最高的前5个 SN 节点
    print(f"\n每个 VN 节点的 Top-5 SN 节点 (按分配概率降序):")
    for i in range(min(3, label.size(0))):
        top_k = min(5, label.size(1))
        top_values, top_indices = torch.topk(label[i], k=top_k)
        print(f"  VN节点 {i}:")
        for rank, (idx, val) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy()), 1):
            print(f"    Top {rank}: SN节点 {idx}, 概率={val:.6f}")

    # 前向传播
    print(f"\n" + "="*60)
    print("执行前向传播...")
    print("="*60)
    model.eval()
    with torch.no_grad():
        output = model(workflow_graph, substrate_graph)
    
    # 打印模型输出
    print(f"\n模型输出信息:")
    print(f"  输出形状: {output.shape}")
    print(f"  输出总和: {output.sum().item():.6f}")
    print(f"  输出最小值: {output.min().item():.6f}")
    print(f"  输出最大值: {output.max().item():.6f}")
    print(f"  输出均值: {output.mean().item():.6f}")
    
    # 打印预测结果（前3行）
    print(f"\n模型输出 (pred) - 前3行:")
    output_np = output.detach().cpu().numpy()
    for i in range(min(3, output.size(0))):
        print(f"  VN节点 {i}: {output_np[i]}")
    
    # 打印标签（前3行）
    print(f"\n标签 (label) - 前3行:")
    for i in range(min(3, label.size(0))):
        print(f"  VN节点 {i}: {label_np[i]}")
    
    # 计算损失
    print(f"\n" + "="*60)
    print("计算测试损失...")
    print("="*60)
    
    # 使用训练时的损失函数
    def kl_mse_loss(output, label):
        eps = 1e-8
        p = torch.clamp(output, min=eps)
        q = torch.clamp(label, min=eps)
        kl_row = torch.sum(q * (torch.log(q) - torch.log(p)), dim=1)
        kl = kl_row.mean()
        mse = torch.mean((output - label) ** 2)
        return mse  # 当前使用 MSE
    
    test_loss = kl_mse_loss(output, label)
    print(f"  测试损失 (MSE): {test_loss.item():.6f}")
    
    # 对比预测与标签的差异
    print(f"\n预测与标签的差异分析:")
    diff = torch.abs(output - label)
    print(f"  平均绝对误差 (MAE): {diff.mean().item():.6f}")
    print(f"  最大绝对误差: {diff.max().item():.6f}")
    print(f"  最小绝对误差: {diff.min().item():.6f}")
    
    # 每个 VN 节点的预测 Top-5 与标签 Top-5 的对比
    print(f"\n每个 VN 节点的预测 Top-5 SN 节点:")
    for i in range(min(3, output.size(0))):
        top_k = min(5, output.size(1))
        pred_top_values, pred_top_indices = torch.topk(output[i], k=top_k)
        label_top_values, label_top_indices = torch.topk(label[i], k=top_k)
        
        print(f"\n  VN节点 {i}:")
        print(f"    预测 Top-5:")
        for rank, (idx, val) in enumerate(zip(pred_top_indices.cpu().numpy(), pred_top_values.cpu().numpy()), 1):
            print(f"      Top {rank}: SN节点 {idx}, 概率={val:.6f}")
        print(f"    标签 Top-5:")
        for rank, (idx, val) in enumerate(zip(label_top_indices.cpu().numpy(), label_top_values.cpu().numpy()), 1):
            print(f"      Top {rank}: SN节点 {idx}, 概率={val:.6f}")
        
        # 计算 Top-5 的重叠度
        pred_set = set(pred_top_indices.cpu().numpy())
        label_set = set(label_top_indices.cpu().numpy())
        overlap = len(pred_set & label_set)
        print(f"    Top-5 重叠数: {overlap}/5")


if __name__ == '__main__':
    main()

