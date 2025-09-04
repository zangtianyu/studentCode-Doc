"""
CNN文本分类模型 - 基于卷积神经网络的文本分类
参考论文: Convolutional Neural Networks for Sentence Classification (Kim, 2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

from .base_model import BaseTextClassifier, TextClassificationHead

class CNNTextClassifier(BaseTextClassifier):
    """CNN文本分类器"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_classes: int,
                 num_filters: int = 100,
                 filter_sizes: List[int] = [3, 4, 5],
                 dropout: float = 0.5,
                 padding_idx: Optional[int] = None,
                 embedding_layer: Optional[nn.Embedding] = None):
        """
        初始化CNN文本分类器
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            num_classes: 分类类别数
            num_filters: 每种卷积核的数量
            filter_sizes: 卷积核大小列表
            dropout: Dropout比率
            padding_idx: 填充索引
            embedding_layer: 预定义的嵌入层
        """
        super(CNNTextClassifier, self).__init__(
            vocab_size, embedding_dim, num_classes, padding_idx
        )
        
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout
        
        # 词嵌入层
        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size
            )
            for filter_size in filter_sizes
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 分类头部
        total_filters = len(filter_sizes) * num_filters
        self.classifier = TextClassificationHead(
            input_dim=total_filters,
            hidden_dim=total_filters // 2,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 词嵌入 [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # 转置为卷积所需的格式 [batch_size, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # 应用多个卷积核
        conv_outputs = []
        for conv in self.convs:
            # 卷积 + ReLU激活 [batch_size, num_filters, conv_seq_len]
            conv_out = F.relu(conv(embedded))
            
            # 最大池化 [batch_size, num_filters, 1]
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            
            # 压缩维度 [batch_size, num_filters]
            pooled = pooled.squeeze(2)
            
            conv_outputs.append(pooled)
        
        # 拼接所有卷积输出 [batch_size, total_filters]
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout
        dropped = self.dropout(concatenated)
        
        # 分类
        output = self.classifier(dropped)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        base_info = super().get_model_info()
        base_info.update({
            'num_filters': self.num_filters,
            'filter_sizes': self.filter_sizes,
            'dropout_rate': self.dropout_rate,
            'total_conv_filters': len(self.filter_sizes) * self.num_filters
        })
        return base_info

class MultiChannelCNN(BaseTextClassifier):
    """多通道CNN文本分类器"""

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_classes: int,
                 num_filters: int = 100,
                 filter_sizes: List[int] = [3, 4, 5],
                 dropout: float = 0.5,
                 padding_idx: Optional[int] = None,
                 static_embedding: Optional[nn.Embedding] = None,
                 non_static_embedding: Optional[nn.Embedding] = None):
        """
        初始化多通道CNN文本分类器

        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            num_classes: 分类类别数
            num_filters: 每种卷积核的数量
            filter_sizes: 卷积核大小列表
            dropout: Dropout比率
            padding_idx: 填充索引
            static_embedding: 静态嵌入层 (不更新)
            non_static_embedding: 非静态嵌入层 (可更新)
        """
        super(MultiChannelCNN, self).__init__(
            vocab_size, embedding_dim, num_classes, padding_idx
        )

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout

        # 静态嵌入通道 (冻结权重)
        if static_embedding is not None:
            self.static_embedding = static_embedding
            self.static_embedding.weight.requires_grad = False
        else:
            self.static_embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )
            nn.init.uniform_(self.static_embedding.weight, -0.1, 0.1)
            self.static_embedding.weight.requires_grad = False

        # 非静态嵌入通道 (可训练权重)
        if non_static_embedding is not None:
            self.non_static_embedding = non_static_embedding
        else:
            self.non_static_embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )
            nn.init.uniform_(self.non_static_embedding.weight, -0.1, 0.1)

        # 卷积层 (输入通道数为2，因为有两个嵌入通道)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim * 2,  # 两个通道
                out_channels=num_filters,
                kernel_size=filter_size
            )
            for filter_size in filter_sizes
        ])

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 分类头部
        total_filters = len(filter_sizes) * num_filters
        self.classifier = TextClassificationHead(
            input_dim=total_filters,
            hidden_dim=total_filters // 2,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len]

        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 静态嵌入 [batch_size, seq_len, embedding_dim]
        static_embedded = self.static_embedding(x)

        # 非静态嵌入 [batch_size, seq_len, embedding_dim]
        non_static_embedded = self.non_static_embedding(x)

        # 拼接两个通道 [batch_size, seq_len, embedding_dim * 2]
        embedded = torch.cat([static_embedded, non_static_embedded], dim=2)

        # 转置为卷积所需的格式 [batch_size, embedding_dim * 2, seq_len]
        embedded = embedded.transpose(1, 2)

        # 应用多个卷积核
        conv_outputs = []
        for conv in self.convs:
            # 卷积 + ReLU激活
            conv_out = F.relu(conv(embedded))

            # 最大池化
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))

            # 压缩维度
            pooled = pooled.squeeze(2)

            conv_outputs.append(pooled)

        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)

        # Dropout
        dropped = self.dropout(concatenated)

        # 分类
        output = self.classifier(dropped)

        return output
