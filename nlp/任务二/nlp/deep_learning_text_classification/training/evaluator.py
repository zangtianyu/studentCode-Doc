"""
评估器 - 负责模型评估和结果分析
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader

from utils.metrics import MetricsCalculator, print_metrics_summary

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, device: torch.device):
        """
        初始化评估器
        
        Args:
            device: 设备
        """
        self.device = device
        self.results = {}
    
    def evaluate_model(self, 
                      model: torch.nn.Module,
                      data_loader: DataLoader,
                      model_name: str = "Model",
                      class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        评估单个模型
        
        Args:
            model: 模型
            data_loader: 数据加载器
            model_name: 模型名称
            class_names: 类别名称列表
            
        Returns:
            评估结果字典
        """
        model.eval()
        metrics_calculator = MetricsCalculator(num_classes=model.num_classes)
        
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                metrics_calculator.update(predictions, targets, probabilities)
        
        # 计算指标
        metrics = metrics_calculator.compute_metrics()
        metrics['loss'] = total_loss / len(data_loader)
        
        # 获取混淆矩阵和分类报告
        confusion_matrix = metrics_calculator.get_confusion_matrix()
        classification_report = metrics_calculator.get_classification_report()
        
        result = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': confusion_matrix,
            'classification_report': classification_report,
            'model_info': model.get_model_info()
        }
        
        self.results[model_name] = result
        
        # 打印结果
        print(f"\n{model_name} 评估结果:")
        print_metrics_summary(metrics, f"{model_name} Metrics")
        print(f"分类报告:\n{classification_report}")
        
        return result
    
    def compare_models(self, 
                      models: Dict[str, torch.nn.Module],
                      data_loader: DataLoader,
                      class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        比较多个模型
        
        Args:
            models: 模型字典 {名称: 模型}
            data_loader: 数据加载器
            class_names: 类别名称列表
            
        Returns:
            比较结果字典
        """
        print("开始模型比较...")
        print("=" * 50)
        
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\n评估模型: {model_name}")
            result = self.evaluate_model(model, data_loader, model_name, class_names)
            comparison_results[model_name] = result
        
        # 生成比较报告
        self._generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def _generate_comparison_report(self, results: Dict[str, Any]):
        """生成比较报告"""
        print("\n" + "=" * 60)
        print("模型比较报告")
        print("=" * 60)
        
        # 准备比较数据
        model_names = list(results.keys())
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'loss']
        
        print(f"{'模型名称':<20}", end="")
        for metric in metrics_to_compare:
            print(f"{metric.capitalize():<12}", end="")
        print("参数量")
        print("-" * 80)
        
        for model_name in model_names:
            result = results[model_name]
            metrics = result['metrics']
            model_info = result['model_info']
            
            print(f"{model_name:<20}", end="")
            for metric in metrics_to_compare:
                value = metrics.get(metric, 0.0)
                print(f"{value:<12.4f}", end="")
            print(f"{model_info['total_parameters']:,}")
        
        # 找出最佳模型
        best_models = {}
        for metric in metrics_to_compare:
            if metric == 'loss':
                # 损失越小越好
                best_model = min(model_names, key=lambda x: results[x]['metrics'].get(metric, float('inf')))
            else:
                # 其他指标越大越好
                best_model = max(model_names, key=lambda x: results[x]['metrics'].get(metric, 0.0))
            best_models[metric] = best_model
        
        print(f"\n最佳模型:")
        for metric, model_name in best_models.items():
            value = results[model_name]['metrics'].get(metric, 0.0)
            print(f"{metric.capitalize()}: {model_name} ({value:.4f})")
    
    def plot_comparison(self, 
                       results: Optional[Dict[str, Any]] = None,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        绘制模型比较图
        
        Args:
            results: 比较结果 (如果为None，使用self.results)
            save_path: 保存路径
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        if results is None:
            results = self.results
        
        if not results:
            print("没有可用的评估结果")
            return None
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('模型性能比较', fontsize=16)
        
        model_names = list(results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # 准备数据
        metrics_data = {metric: [] for metric in metrics_to_plot}
        for model_name in model_names:
            for metric in metrics_to_plot:
                value = results[model_name]['metrics'].get(metric, 0.0)
                metrics_data[metric].append(value)
        
        # 绘制柱状图
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(model_names, metrics_data[metric])
            ax.set_title(f'{metric.capitalize()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, value in zip(bars, metrics_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较图已保存到: {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self,
                               results: Optional[Dict[str, Any]] = None,
                               class_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        绘制混淆矩阵对比
        
        Args:
            results: 比较结果
            class_names: 类别名称
            save_path: 保存路径
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        if results is None:
            results = self.results
        
        if not results:
            print("没有可用的评估结果")
            return None
        
        model_names = list(results.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            cm = results[model_name]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=class_names or range(cm.shape[1]),
                       yticklabels=class_names or range(cm.shape[0]))
            
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵图已保存到: {save_path}")
        
        return fig
    
    def save_results(self, 
                    results: Optional[Dict[str, Any]] = None,
                    save_path: str = "results/evaluation_results.txt"):
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果
            save_path: 保存路径
        """
        if results is None:
            results = self.results
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("模型评估结果报告\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"模型: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                # 写入指标
                metrics = result['metrics']
                for metric, value in metrics.items():
                    if isinstance(value, (list, np.ndarray)):
                        continue
                    f.write(f"{metric}: {value:.4f}\n")
                
                # 写入模型信息
                model_info = result['model_info']
                f.write(f"参数量: {model_info['total_parameters']:,}\n")
                f.write(f"模型大小: {model_info['model_size_mb']:.2f} MB\n")
                
                # 写入分类报告
                f.write(f"\n分类报告:\n{result['classification_report']}\n")
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"评估结果已保存到: {save_path}")
