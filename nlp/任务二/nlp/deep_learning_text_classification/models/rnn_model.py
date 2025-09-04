"""
RNN文本分类模型 - 基于循环神经网络的文本分类
支持LSTM和GRU，以及双向RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from .base_model import BaseTextClassifier, TextClassificationHead

class RNNTextClassifier(BaseTextClassifier):
    """RNN文本分类器"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 rnn_type: str = 'LSTM',
                 bidirectional: bool = True,
                 dropout: float = 0.5,
                 padding_idx: Optional[int] = None,
                 embedding_layer: Optional[nn.Embedding] = None):
        """
        初始化RNN文本分类器
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            num_classes: 分类类别数
            hidden_dim: RNN隐藏层维度
            num_layers: RNN层数
            rnn_type: RNN类型 ('LSTM' 或 'GRU')
            bidirectional: 是否使用双向RNN
            dropout: Dropout比率
            padding_idx: 填充索引
            embedding_layer: 预定义的嵌入层
        """
        super(RNNTextClassifier, self).__init__(
            vocab_size, embedding_dim, num_classes, padding_idx
        )
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        
        # 词嵌入层
        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # RNN层
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 计算RNN输出维度
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # 分类头部
        self.classifier = TextClassificationHead(
            input_dim=rnn_output_dim,
            hidden_dim=rnn_output_dim // 2,
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
        
        # RNN处理 [batch_size, seq_len, hidden_dim * directions]
        rnn_output, _ = self.rnn(embedded)
        
        # 取最后一个时间步的输出 [batch_size, hidden_dim * directions]
        if self.bidirectional:
            # 对于双向RNN，取前向和后向的最后输出
            forward_output = rnn_output[:, -1, :self.hidden_dim]
            backward_output = rnn_output[:, 0, self.hidden_dim:]
            last_output = torch.cat([forward_output, backward_output], dim=1)
        else:
            last_output = rnn_output[:, -1, :]
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # 分类
        output = self.classifier(dropped)
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        base_info = super().get_model_info()
        base_info.update({
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'rnn_type': self.rnn_type,
            'bidirectional': self.bidirectional,
            'dropout_rate': self.dropout_rate,
            'rnn_output_dim': self.hidden_dim * (2 if self.bidirectional else 1)
        })
        return base_info

class AttentionRNN(BaseTextClassifier):
    """带注意力机制的RNN文本分类器"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 rnn_type: str = 'LSTM',
                 bidirectional: bool = True,
                 dropout: float = 0.5,
                 attention_dim: int = 64,
                 padding_idx: Optional[int] = None,
                 embedding_layer: Optional[nn.Embedding] = None):
        """
        初始化带注意力的RNN文本分类器
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            num_classes: 分类类别数
            hidden_dim: RNN隐藏层维度
            num_layers: RNN层数
            rnn_type: RNN类型
            bidirectional: 是否使用双向RNN
            dropout: Dropout比率
            attention_dim: 注意力维度
            padding_idx: 填充索引
            embedding_layer: 预定义的嵌入层
        """
        super(AttentionRNN, self).__init__(
            vocab_size, embedding_dim, num_classes, padding_idx
        )
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        self.attention_dim = attention_dim
        
        # 词嵌入层
        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # RNN层
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # 注意力机制
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = SelfAttention(rnn_output_dim, attention_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 分类头部
        self.classifier = TextClassificationHead(
            input_dim=rnn_output_dim,
            hidden_dim=rnn_output_dim // 2,
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
        
        # RNN处理 [batch_size, seq_len, hidden_dim * directions]
        rnn_output, _ = self.rnn(embedded)
        
        # 注意力机制 [batch_size, hidden_dim * directions]
        attended_output = self.attention(rnn_output)
        
        # Dropout
        dropped = self.dropout(attended_output)
        
        # 分类
        output = self.classifier(dropped)
        
        return output

class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, input_dim: int, attention_dim: int):
        """
        初始化自注意力机制
        
        Args:
            input_dim: 输入维度
            attention_dim: 注意力维度
        """
        super(SelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # 注意力权重计算
        self.attention_weights = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.xavier_uniform_(self.context_vector.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            加权平均后的张量 [batch_size, input_dim]
        """
        # 计算注意力分数 [batch_size, seq_len, attention_dim]
        attention_scores = torch.tanh(self.attention_weights(x))
        
        # 计算注意力权重 [batch_size, seq_len, 1]
        attention_weights = self.context_vector(attention_scores)
        
        # 应用softmax [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权平均 [batch_size, input_dim]
        weighted_output = torch.sum(x * attention_weights, dim=1)
        
        return weighted_output
