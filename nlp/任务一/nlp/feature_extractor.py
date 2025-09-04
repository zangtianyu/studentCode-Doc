import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import re

class BagOfWordsExtractor:
    def __init__(self, max_features: int = 5000, min_df: int = 1, max_df: float = 1.0):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
        self.feature_names = []
        self.document_frequency = {}
        self.n_documents = 0
        
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def build_vocabulary(self, texts: List[str]):
        word_counts = Counter()
        doc_word_counts = defaultdict(int)
        
        self.n_documents = len(texts)
        
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)
            
            unique_words = set(words)
            for word in unique_words:
                doc_word_counts[word] += 1
        
        self.document_frequency = dict(doc_word_counts)
        
        filtered_words = []
        for word, count in word_counts.items():
            df = self.document_frequency[word]
            if df >= self.min_df and df <= self.max_df * self.n_documents:
                filtered_words.append((word, count))
        
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        if self.max_features:
            filtered_words = filtered_words[:self.max_features]
        
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(filtered_words)}
        self.feature_names = [word for word, _ in filtered_words]
        
        print(f"构建词汇表完成，共 {len(self.vocabulary)} 个词汇")
        
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.vocabulary:
            raise ValueError("词汇表未构建，请先调用 build_vocabulary")
        
        features = np.zeros((len(texts), len(self.vocabulary)))
        
        for i, text in enumerate(texts):
            words = self.tokenize(text)
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    features[i, self.vocabulary[word]] = count
                    
        return features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.build_vocabulary(texts)
        return self.transform(texts)

class NGramExtractor:
    def __init__(self, n: int = 2, max_features: int = 5000, min_df: int = 1, max_df: float = 1.0):
        self.n = n
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
        self.feature_names = []
        self.document_frequency = {}
        self.n_documents = 0
        
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def get_ngrams(self, words: List[str]) -> List[str]:
        if len(words) < self.n:
            return []
        
        ngrams = []
        for i in range(len(words) - self.n + 1):
            ngram = ' '.join(words[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def build_vocabulary(self, texts: List[str]):
        ngram_counts = Counter()
        doc_ngram_counts = defaultdict(int)
        
        self.n_documents = len(texts)
        
        for text in texts:
            words = self.tokenize(text)
            ngrams = self.get_ngrams(words)
            ngram_counts.update(ngrams)
            
            unique_ngrams = set(ngrams)
            for ngram in unique_ngrams:
                doc_ngram_counts[ngram] += 1
        
        self.document_frequency = dict(doc_ngram_counts)
        
        filtered_ngrams = []
        for ngram, count in ngram_counts.items():
            df = self.document_frequency[ngram]
            if df >= self.min_df and df <= self.max_df * self.n_documents:
                filtered_ngrams.append((ngram, count))
        
        filtered_ngrams.sort(key=lambda x: x[1], reverse=True)
        
        if self.max_features:
            filtered_ngrams = filtered_ngrams[:self.max_features]
        
        self.vocabulary = {ngram: idx for idx, (ngram, _) in enumerate(filtered_ngrams)}
        self.feature_names = [ngram for ngram, _ in filtered_ngrams]
        
        print(f"构建{self.n}-gram词汇表完成，共 {len(self.vocabulary)} 个特征")
        
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.vocabulary:
            raise ValueError("词汇表未构建，请先调用 build_vocabulary")
        
        features = np.zeros((len(texts), len(self.vocabulary)))
        
        for i, text in enumerate(texts):
            words = self.tokenize(text)
            ngrams = self.get_ngrams(words)
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if ngram in self.vocabulary:
                    features[i, self.vocabulary[ngram]] = count
                    
        return features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.build_vocabulary(texts)
        return self.transform(texts)

class CombinedFeatureExtractor:
    def __init__(self, use_bow: bool = True, use_bigram: bool = True, use_trigram: bool = False,
                 max_features_bow: int = 3000, max_features_ngram: int = 2000):
        self.use_bow = use_bow
        self.use_bigram = use_bigram
        self.use_trigram = use_trigram
        
        self.extractors = {}
        
        if use_bow:
            self.extractors['bow'] = BagOfWordsExtractor(max_features=max_features_bow)
        if use_bigram:
            self.extractors['bigram'] = NGramExtractor(n=2, max_features=max_features_ngram)
        if use_trigram:
            self.extractors['trigram'] = NGramExtractor(n=3, max_features=max_features_ngram)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        features_list = []
        
        for name, extractor in self.extractors.items():
            print(f"提取 {name} 特征...")
            features = extractor.fit_transform(texts)
            features_list.append(features)
        
        if features_list:
            combined_features = np.hstack(features_list)
            print(f"组合特征维度: {combined_features.shape}")
            return combined_features
        else:
            raise ValueError("至少需要启用一种特征提取方法")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        features_list = []
        
        for name, extractor in self.extractors.items():
            features = extractor.transform(texts)
            features_list.append(features)
        
        if features_list:
            return np.hstack(features_list)
        else:
            raise ValueError("至少需要启用一种特征提取方法")
    
    def get_feature_names(self) -> List[str]:
        feature_names = []
        for name, extractor in self.extractors.items():
            names = [f"{name}_{fname}" for fname in extractor.feature_names]
            feature_names.extend(names)
        return feature_names

class FeatureSelector:
    def __init__(self, method: str = 'chi2', k: int = 1000):
        self.method = method
        self.k = k
        self.selected_features = None
        self.feature_scores = None
        
    def chi2_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        
        for i in range(n_features):
            feature = X[:, i]
            
            pos_with_feature = np.sum((feature > 0) & (y == 1))
            pos_without_feature = np.sum((feature == 0) & (y == 1))
            neg_with_feature = np.sum((feature > 0) & (y == 0))
            neg_without_feature = np.sum((feature == 0) & (y == 0))
            
            observed = np.array([[pos_with_feature, pos_without_feature],
                               [neg_with_feature, neg_without_feature]])
            
            row_sums = observed.sum(axis=1)
            col_sums = observed.sum(axis=0)
            total = observed.sum()
            
            expected = np.outer(row_sums, col_sums) / total
            
            chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-10))
            scores[i] = chi2
            
        return scores
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.method == 'chi2':
            self.feature_scores = self.chi2_score(X, y)
        else:
            raise ValueError(f"不支持的特征选择方法: {self.method}")
        
        top_k_indices = np.argsort(self.feature_scores)[-self.k:]
        self.selected_features = sorted(top_k_indices)
        
        print(f"特征选择完成，从 {X.shape[1]} 个特征中选择了 {len(self.selected_features)} 个")
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features is None:
            raise ValueError("特征选择器未训练，请先调用 fit")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

if __name__ == "__main__":
    texts = [
        "this movie is great and amazing",
        "terrible film with bad acting",
        "excellent story and wonderful cast",
        "boring movie with poor direction"
    ]
    
    print("测试Bag-of-Words特征提取:")
    bow = BagOfWordsExtractor(max_features=10)
    bow_features = bow.fit_transform(texts)
    print(f"特征矩阵形状: {bow_features.shape}")
    print(f"词汇表: {bow.feature_names}")
    
    print("\n测试N-gram特征提取:")
    bigram = NGramExtractor(n=2, max_features=10)
    bigram_features = bigram.fit_transform(texts)
    print(f"特征矩阵形状: {bigram_features.shape}")
    print(f"Bigram特征: {bigram.feature_names}")
    
    print("\n测试组合特征提取:")
    combined = CombinedFeatureExtractor(max_features_bow=5, max_features_ngram=5)
    combined_features = combined.fit_transform(texts)
    print(f"组合特征矩阵形状: {combined_features.shape}")
