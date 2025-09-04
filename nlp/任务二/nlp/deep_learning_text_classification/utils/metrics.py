"""
评估指标计算
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, num_classes: int = 2):
        """
        初始化评估指标计算器
        
        Args:
            num_classes: 类别数量
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置累积的预测结果"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """
        更新预测结果
        
        Args:
            predictions: 预测标签
            targets: 真实标签
            probabilities: 预测概率 (可选)
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        
        self.all_predictions.extend(predictions.flatten())
        self.all_targets.extend(targets.flatten())
        
        if probabilities is not None:
            if probabilities.ndim == 1:
                # 二分类情况
                self.all_probabilities.extend(probabilities.flatten())
            else:
                # 多分类情况，取最大概率
                self.all_probabilities.extend(np.max(probabilities, axis=1))
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Returns:
            包含各种指标的字典
        """
        if not self.all_predictions or not self.all_targets:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # 基本指标
        accuracy = accuracy_score(targets, predictions)
        
        # 精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # 各类别的精确率、召回率、F1分数
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist()
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        获取混淆矩阵
        
        Returns:
            混淆矩阵
        """
        if not self.all_predictions or not self.all_targets:
            return np.array([])
        
        return confusion_matrix(self.all_targets, self.all_predictions)
    
    def get_classification_report(self) -> str:
        """
        获取分类报告
        
        Returns:
            分类报告字符串
        """
        if not self.all_predictions or not self.all_targets:
            return ""
        
        return classification_report(self.all_targets, self.all_predictions)
    
    def plot_confusion_matrix(self, class_names: Optional[List[str]] = None, 
                            save_path: Optional[str] = None, 
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        绘制混淆矩阵
        
        Args:
            class_names: 类别名称列表
            save_path: 保存路径
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        cm = self.get_confusion_matrix()
        
        if cm.size == 0:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用seaborn绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names or range(cm.shape[1]),
                   yticklabels=class_names or range(cm.shape[0]))
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测结果
        targets: 真实标签
        
    Returns:
        准确率
    """
    correct = (predictions == targets).float()
    return correct.mean().item()

def calculate_loss_and_accuracy(model, data_loader, criterion, device):
    """
    计算模型在数据集上的损失和准确率
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        平均损失和准确率
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # 计算准确率
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def print_metrics_summary(metrics: Dict[str, float], title: str = "Metrics Summary"):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            continue  # 跳过列表类型的指标
        print(f"{key.capitalize()}: {value:.4f}")
    
    print()

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数 (越高越好)
            model: 模型
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model: torch.nn.Module):
        """保存检查点"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
