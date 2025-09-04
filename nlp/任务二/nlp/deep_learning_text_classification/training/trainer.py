"""
训练器 - 负责模型训练和验证
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from tqdm import tqdm

from utils.metrics import MetricsCalculator, EarlyStopping, calculate_loss_and_accuracy

class Trainer:
    """模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            device: 设备
            config: 训练配置
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 5),
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # 评估器
        self.metrics_calculator = MetricsCalculator(num_classes=model.num_classes)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_name = self.config.get('optimizer', 'Adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_name = self.config.get('scheduler', None)
        
        if scheduler_name == 'StepLR':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'ReduceLROnPlateau':
            patience = self.config.get('scheduler_patience', 3)
            factor = self.config.get('scheduler_factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=patience, factor=factor, verbose=True
            )
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        self.metrics_calculator.reset()
        
        progress_bar = tqdm(self.train_loader, desc="训练中")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            self.metrics_calculator.update(predictions, targets, probabilities)
            
            # 更新进度条
            if batch_idx % self.config.get('log_interval', 100) == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Batch': f'{batch_idx}/{len(self.train_loader)}'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics_calculator.compute_metrics()
        accuracy = metrics.get('accuracy', 0.0)
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="验证中"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                self.metrics_calculator.update(predictions, targets, probabilities)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calculator.compute_metrics()
        accuracy = metrics.get('accuracy', 0.0)
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """完整的训练流程"""
        epochs = self.config.get('epochs', 20)
        save_best_model = self.config.get('save_best_model', True)
        save_model_path = self.config.get('save_model_path', 'results/models')
        
        print(f"开始训练，共 {epochs} 个epoch")
        print("=" * 50)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 打印结果
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s")
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 30)
            
            # 保存最佳模型
            if save_best_model and val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_model_path:
                    os.makedirs(save_model_path, exist_ok=True)
                    model_path = os.path.join(save_model_path, 'best_model.pth')
                    self.model.save_model(model_path, save_optimizer=True, optimizer=self.optimizer)
                    print(f"保存最佳模型: {model_path}")
            
            # 早停检查
            if self.early_stopping(val_acc, self.model):
                print(f"早停触发，在第 {epoch+1} 个epoch停止训练")
                break
        
        total_time = time.time() - start_time
        print(f"\n训练完成，总用时: {total_time:.2f}秒")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
        
        return {
            'best_val_accuracy': best_val_acc,
            'total_time': total_time,
            'epochs_trained': epoch + 1,
            'history': self.history
        }
    
    def evaluate(self, data_loader: DataLoader = None) -> Dict[str, Any]:
        """评估模型"""
        if data_loader is None:
            data_loader = self.test_loader
        
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="评估中"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                self.metrics_calculator.update(predictions, targets, probabilities)
        
        avg_loss = total_loss / len(data_loader)
        metrics = self.metrics_calculator.compute_metrics()
        
        # 添加损失到指标中
        metrics['loss'] = avg_loss
        
        return metrics
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader, desc="预测中"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
