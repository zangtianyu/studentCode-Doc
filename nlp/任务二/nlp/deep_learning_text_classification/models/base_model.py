"""
基础模型类 - 定义所有文本分类模型的通用接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTextClassifier(nn.Module, ABC):
    """文本分类模型基类"""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 num_classes: int,
                 padding_idx: Optional[int] = None):
        """
        初始化基础模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            num_classes: 分类类别数
            padding_idx: 填充索引
        """
        super(BaseTextClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        
        # 子类需要实现的组件
        self.embedding = None
        self.classifier = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print(f"\n{info['model_name']} 模型信息:")
        print("=" * 40)
        print(f"词汇表大小: {info['vocab_size']:,}")
        print(f"词嵌入维度: {info['embedding_dim']}")
        print(f"分类类别数: {info['num_classes']}")
        print(f"总参数量: {info['total_parameters']:,}")
        print(f"可训练参数: {info['trainable_parameters']:,}")
        print(f"模型大小: {info['model_size_mb']:.2f} MB")
        print("=" * 40)
    
    def freeze_embedding(self):
        """冻结词嵌入层"""
        if self.embedding is not None:
            for param in self.embedding.parameters():
                param.requires_grad = False
            print("词嵌入层已冻结")
    
    def unfreeze_embedding(self):
        """解冻词嵌入层"""
        if self.embedding is not None:
            for param in self.embedding.parameters():
                param.requires_grad = True
            print("词嵌入层已解冻")
    
    def get_embedding_weights(self) -> torch.Tensor:
        """获取词嵌入权重"""
        if self.embedding is not None:
            return self.embedding.weight.data
        else:
            raise ValueError("模型没有嵌入层")
    
    def set_embedding_weights(self, weights: torch.Tensor):
        """设置词嵌入权重"""
        if self.embedding is not None:
            if weights.shape != self.embedding.weight.shape:
                raise ValueError(f"权重形状不匹配: {weights.shape} vs {self.embedding.weight.shape}")
            self.embedding.weight.data.copy_(weights)
            print("词嵌入权重已更新")
        else:
            raise ValueError("模型没有嵌入层")
    
    def save_model(self, filepath: str, save_optimizer: bool = False, optimizer=None):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            save_optimizer: 是否保存优化器状态
            optimizer: 优化器对象
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'num_classes': self.num_classes,
                'padding_idx': self.padding_idx
            },
            'model_info': self.get_model_info()
        }
        
        if save_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: torch.device = None):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            device: 设备
            
        Returns:
            加载的模型实例
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        model_config = checkpoint['model_config']
        
        # 创建模型实例 (子类需要实现)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"模型已从 {filepath} 加载")
        return model, checkpoint.get('optimizer_state_dict', None)

class TextClassificationHead(nn.Module):
    """文本分类头部"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 dropout: float = 0.5,
                 activation: str = 'relu'):
        """
        初始化分类头部
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
            dropout: Dropout比率
            activation: 激活函数类型
        """
        super(TextClassificationHead, self).__init__()
        
        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        return self.classifier(x)
