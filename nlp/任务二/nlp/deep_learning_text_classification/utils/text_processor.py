"""
文本预处理工具 - 使用简单分词器避免NLTK问题
"""

import re
import string
from typing import List, Optional
import pandas as pd

# 使用简单分词器，避免NLTK依赖问题
NLTK_AVAILABLE = False
print("使用简单分词器 (避免NLTK依赖问题)")

class TextProcessor:
    """文本预处理器"""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = False,
                 min_length: int = 1):
        """
        初始化文本预处理器
        
        Args:
            lowercase: 是否转换为小写
            remove_punctuation: 是否移除标点符号
            remove_numbers: 是否移除数字
            remove_stopwords: 是否移除停用词
            min_length: 最小词长度
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """
        清洗单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 转换为小写
        if self.lowercase:
            text = text.lower()
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        if not text:
            return []

        # 使用简单分词
        tokens = self._simple_tokenize(text)
        
        # 过滤处理
        filtered_tokens = []
        for token in tokens:
            # 移除标点符号
            if self.remove_punctuation and token in string.punctuation:
                continue
            
            # 移除数字
            if self.remove_numbers and token.isdigit():
                continue
            
            # 移除停用词
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # 检查最小长度
            if len(token) < self.min_length:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens

    def _simple_tokenize(self, text: str) -> List[str]:
        """
        简单分词器 - 使用正则表达式分词

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        # 使用正则表达式进行简单分词
        # 匹配字母数字组合的单词
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def process_text(self, text: str) -> List[str]:
        """
        完整的文本处理流程
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的词列表
        """
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        # 分词
        tokens = self.tokenize(cleaned_text)
        
        return tokens
    
    def process_texts(self, texts: List[str]) -> List[List[str]]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            处理后的词列表的列表
        """
        return [self.process_text(text) for text in texts]

def create_text_processor(config: Optional[dict] = None) -> TextProcessor:
    """
    创建文本处理器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        TextProcessor实例
    """
    if config is None:
        config = {}
    
    return TextProcessor(
        lowercase=config.get('lowercase', True),
        remove_punctuation=config.get('remove_punctuation', True),
        remove_numbers=config.get('remove_numbers', False),
        remove_stopwords=config.get('remove_stopwords', False),
        min_length=config.get('min_length', 1)
    )
