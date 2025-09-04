"""
词汇表构建和管理
"""

from typing import List, Dict, Optional, Tuple
from collections import Counter
import pickle
import os

class Vocabulary:
    """词汇表类"""
    
    def __init__(self, 
                 max_size: Optional[int] = None,
                 min_freq: int = 1,
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>'):
        """
        初始化词汇表
        
        Args:
            max_size: 最大词汇表大小
            min_freq: 最小词频
            pad_token: 填充标记
            unk_token: 未知词标记
        """
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # 词汇表映射
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # 特殊标记
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """添加特殊标记"""
        special_tokens = [self.pad_token, self.unk_token]
        for token in special_tokens:
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
    
    def build_from_texts(self, texts: List[List[str]]):
        """
        从文本列表构建词汇表
        
        Args:
            texts: 分词后的文本列表
        """
        # 统计词频
        for text in texts:
            self.word_freq.update(text)
        
        # 按频率排序
        sorted_words = self.word_freq.most_common()
        
        # 过滤低频词
        filtered_words = [(word, freq) for word, freq in sorted_words 
                         if freq >= self.min_freq]
        
        # 限制词汇表大小
        if self.max_size is not None:
            # 保留空间给特殊标记
            max_words = self.max_size - len(self.word2idx)
            filtered_words = filtered_words[:max_words]
        
        # 构建映射
        for word, freq in filtered_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text: List[str]) -> List[int]:
        """
        将词列表编码为索引列表
        
        Args:
            text: 词列表
            
        Returns:
            索引列表
        """
        unk_idx = self.word2idx[self.unk_token]
        return [self.word2idx.get(word, unk_idx) for word in text]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        将索引列表解码为词列表
        
        Args:
            indices: 索引列表
            
        Returns:
            词列表
        """
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        """检查词是否在词汇表中"""
        return word in self.word2idx
    
    def get_word_index(self, word: str) -> int:
        """获取词的索引"""
        return self.word2idx.get(word, self.word2idx[self.unk_token])
    
    def get_index_word(self, index: int) -> str:
        """获取索引对应的词"""
        return self.idx2word.get(index, self.unk_token)
    
    def save(self, filepath: str):
        """
        保存词汇表
        
        Args:
            filepath: 保存路径
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'max_size': self.max_size,
            'min_freq': self.min_freq,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """
        加载词汇表
        
        Args:
            filepath: 文件路径
            
        Returns:
            Vocabulary实例
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        vocab = cls(
            max_size=vocab_data['max_size'],
            min_freq=vocab_data['min_freq'],
            pad_token=vocab_data['pad_token'],
            unk_token=vocab_data['unk_token']
        )
        
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = vocab_data['idx2word']
        vocab.word_freq = Counter(vocab_data['word_freq'])
        
        return vocab
    
    def get_stats(self) -> Dict:
        """获取词汇表统计信息"""
        return {
            'vocab_size': len(self),
            'total_words': sum(self.word_freq.values()),
            'unique_words': len(self.word_freq),
            'avg_freq': sum(self.word_freq.values()) / len(self.word_freq) if self.word_freq else 0,
            'most_common': self.word_freq.most_common(10)
        }

def pad_sequences(sequences: List[List[int]], 
                 max_length: int,
                 pad_value: int = 0,
                 truncate: str = 'post',
                 pad: str = 'post') -> List[List[int]]:
    """
    填充序列到固定长度
    
    Args:
        sequences: 序列列表
        max_length: 最大长度
        pad_value: 填充值
        truncate: 截断方式 ('pre' 或 'post')
        pad: 填充方式 ('pre' 或 'post')
        
    Returns:
        填充后的序列列表
    """
    padded_sequences = []
    
    for seq in sequences:
        # 截断
        if len(seq) > max_length:
            if truncate == 'pre':
                seq = seq[-max_length:]
            else:  # 'post'
                seq = seq[:max_length]
        
        # 填充
        if len(seq) < max_length:
            pad_length = max_length - len(seq)
            if pad == 'pre':
                seq = [pad_value] * pad_length + seq
            else:  # 'post'
                seq = seq + [pad_value] * pad_length
        
        padded_sequences.append(seq)
    
    return padded_sequences
