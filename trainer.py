import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

from model import SimuVNE
from dataset import create_data_loaders


class Trainer:
    """训练器类"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 创建模型
        self.model = SimuVNE(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 64),
            hist_dim=config.get('hist_dim', 32)
        ).to(self.device)
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()  # 二分类损失
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 50),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.processor = create_data_loaders(
            data_dir=config.get('data_dir', './data'),
            batch_size=config.get('batch_size', 32),
            train_ratio=config.get('train_ratio', 0.8),
            hist_dim=config.get('hist_dim', 32),
            normalize_features=config.get('normalize_features', True)
        )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # 创建输出目录
        self.output_dir = config.get('output_dir', './outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TensorBoard
        if config.get('use_tensorboard', True):
            log_dir = os.path.join(self.output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        print(f"训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_loss = 0.0
            batch_samples = 0
            
            # 由于每个样本的图大小不同，需要逐个处理
            for i in range(len(batch['graphs_i'])):
                self.optimizer.zero_grad()
                
                # 获取单个样本
                graph_i = batch['graphs_i'][i].to(self.device)
                graph_j = batch['graphs_j'][i].to(self.device)
                label = batch['labels'][i].to(self.device)
                
                # 前向传播（hist(S)在模型内部自动计算）
                output = self.model(graph_i, graph_j)
                
                # 计算损失
                loss = self.criterion(output, label)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizer.step()
                
                batch_loss += loss.item()
                batch_samples += 1
            
            if batch_samples > 0:
                avg_batch_loss = batch_loss / batch_samples
                total_loss += avg_batch_loss
                num_batches += 1
            
            # 打印进度
            if batch_idx % self.config.get('log_interval', 10) == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {avg_batch_loss:.6f}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch_loss = 0.0
                batch_correct = 0
                batch_samples = 0
                
                # 逐个处理样本
                for i in range(len(batch['graphs_i'])):
                    graph_i = batch['graphs_i'][i].to(self.device)
                    graph_j = batch['graphs_j'][i].to(self.device)
                    label = batch['labels'][i].to(self.device)
                    
                    # 前向传播（hist(S)在模型内部自动计算）
                    output = self.model(graph_i, graph_j)
                    
                    # 计算损失
                    loss = self.criterion(output, label)
                    batch_loss += loss.item()
                    
                    # 计算准确率
                    pred = torch.sigmoid(output) > 0.5
                    correct = (pred == label).float().sum().item()
                    total_elements = label.numel()
                    
                    batch_correct += correct
                    batch_samples += total_elements
                
                if batch_samples > 0:
                    total_loss += batch_loss / len(batch['graphs_i'])
                    total_correct += batch_correct
                    total_samples += batch_samples
                    num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"从 {checkpoint_path} 加载检查点，epoch: {checkpoint['epoch']}")
        return True
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """主训练循环"""
        start_epoch = 0
        
        # 如果指定了恢复路径，加载检查点
        if resume_from:
            if self.load_checkpoint(resume_from):
                start_epoch = len(self.train_losses)
        
        print(f"\n开始训练，共 {num_epochs} 个epoch")
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 检查是否是最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
            
            # 保存检查点
            if (epoch + 1) % self.config.get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  训练损失: {train_loss:.6f}')
            print(f'  验证损失: {val_loss:.6f}')
            print(f'  验证准确率: {val_acc:.4f}')
            print(f'  学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  用时: {epoch_time:.2f}s')
            print(f'  最佳验证损失: {self.best_val_loss:.6f}')
            print(f'  最佳验证准确率: {self.best_val_acc:.4f}')
            print('-' * 50)
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.2f}s")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if self.writer:
            self.writer.close()


def create_trainer(config_path: str = None, **kwargs) -> Trainer:
    """创建训练器实例"""
    
    # 默认配置
    default_config = {
        'input_dim': 6,
        'hidden_dim': 64,
        'hist_dim': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 16,  # 由于需要逐个处理，使用较小的批大小
        'train_ratio': 0.8,
        'lr_step_size': 50,
        'lr_gamma': 0.5,
        'data_dir': '/home/zrz/SimuVNE/data',
        'output_dir': '/home/zrz/SimuVNE/outputs',
        'normalize_features': True,
        'use_tensorboard': True,
        'log_interval': 5,
        'save_interval': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 如果提供了配置文件，加载配置
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        default_config.update(file_config)
    
    # 更新配置
    default_config.update(kwargs)
    
    return Trainer(default_config)


if __name__ == "__main__":
    # 创建训练器
    trainer = create_trainer()
    
    # 开始训练
    trainer.train(num_epochs=100)
