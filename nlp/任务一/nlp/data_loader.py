import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import Counter
import urllib.request
import zipfile
import pandas as pd

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
    def download_rotten_tomatoes_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
        filename = os.path.join(self.data_dir, "rt-polaritydata.tar.gz")
        
        if not os.path.exists(filename):
            print("正在下载Rotten Tomatoes数据集...")
            urllib.request.urlretrieve(url, filename)
            print("下载完成")
            
        import tarfile
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(self.data_dir)
            
    def load_movie_reviews(self):
        pos_file = os.path.join(self.data_dir, "rt-polaritydata", "rt-polarity.pos")
        neg_file = os.path.join(self.data_dir, "rt-polaritydata", "rt-polarity.neg")

        texts = []
        labels = []

        if os.path.exists(pos_file):
            with open(pos_file, 'r', encoding='latin-1') as f:
                for line in f:
                    texts.append(line.strip())
                    labels.append(1)

        if os.path.exists(neg_file):
            with open(neg_file, 'r', encoding='latin-1') as f:
                for line in f:
                    texts.append(line.strip())
                    labels.append(0)

        return texts, labels

    def load_sentiment_analysis_data(self):
        train_file = os.path.join(self.data_dir, "train.tsv", "train.tsv")
        test_file = os.path.join(self.data_dir, "test.tsv", "test.tsv")

        if not os.path.exists(train_file):
            train_file = os.path.join(self.data_dir, "train.tsv")
        if not os.path.exists(test_file):
            test_file = os.path.join(self.data_dir, "test.tsv")

        texts = []
        labels = []

        if os.path.exists(train_file):
            print(f"加载训练数据: {train_file}")
            df_train = pd.read_csv(train_file, sep='\t', encoding='utf-8')

            print(f"训练数据形状: {df_train.shape}")
            print(f"训练数据列: {df_train.columns.tolist()}")

            df_train = df_train.dropna(subset=['Phrase', 'Sentiment'])

            for _, row in df_train.iterrows():
                phrase = row['Phrase']
                sentiment = row['Sentiment']

                if pd.isna(phrase) or pd.isna(sentiment):
                    continue

                if not isinstance(phrase, str):
                    phrase = str(phrase)

                if phrase.strip():
                    texts.append(phrase.strip())
                    labels.append(int(sentiment))

            print(f"成功加载训练数据: {len(texts)} 条样本")

        if os.path.exists(test_file):
            print(f"加载测试数据: {test_file}")
            df_test = pd.read_csv(test_file, sep='\t', encoding='utf-8')

            print(f"测试数据形状: {df_test.shape}")
            print(f"测试数据列: {df_test.columns.tolist()}")

            df_test = df_test.dropna(subset=['Phrase'])

            test_texts = []
            for _, row in df_test.iterrows():
                phrase = row['Phrase']

                if pd.isna(phrase):
                    continue

                if not isinstance(phrase, str):
                    phrase = str(phrase)

                if phrase.strip():
                    test_texts.append(phrase.strip())

            print(f"成功加载测试数据: {len(test_texts)} 条样本")

        return texts, labels


    
    def load_sentiment_dataset(self):
        """
        Load the sentiment analysis dataset from the data folder.
        The dataset contains train.tsv and test.tsv files.
        """
        train_file = os.path.join(self.data_dir, "train.tsv", "train.tsv")
        test_file = os.path.join(self.data_dir, "test.tsv", "test.tsv")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Dataset files not found in {self.data_dir}")
        
        # Load training data
        train_df = pd.read_csv(train_file, sep='\t')
        
        # Load test data
        test_df = pd.read_csv(test_file, sep='\t')
        
        print(f"成功加载数据集:")
        print(f"训练集: {len(train_df)} 样本")
        print(f"测试集: {len(test_df)} 样本")
        
        # Extract texts and labels from training data
        train_texts = train_df['Phrase'].tolist()
        train_labels = train_df['Sentiment'].tolist()
        
        # Extract texts from test data (no labels)
        test_texts = test_df['Phrase'].tolist()
        test_labels = []  # No labels for test data
        
        return train_texts, train_labels, test_texts, test_labels
    
    def create_sample_data(self):
        positive_samples = [
            "This movie is absolutely fantastic and amazing",
            "Great acting and wonderful storyline",
            "Excellent cinematography and brilliant direction",
            "Outstanding performance by all actors",
            "Beautiful and touching story",
            "Incredible visual effects and sound",
            "Perfect blend of action and emotion",
            "Masterpiece of modern cinema",
            "Brilliant script and amazing cast",
            "Wonderful entertainment for the whole family"
        ]
        
        negative_samples = [
            "This movie is terrible and boring",
            "Poor acting and weak storyline",
            "Bad cinematography and awful direction",
            "Disappointing performance by actors",
            "Boring and predictable story",
            "Terrible visual effects and sound",
            "Poor blend of action and drama",
            "Waste of time and money",
            "Weak script and bad casting",
            "Awful entertainment experience"
        ]
        
        texts = positive_samples + negative_samples
        labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        
        return texts, labels
    
    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_data(self, texts: List[str], labels: List[int], 
                   train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15, shuffle: bool = True):
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("数据集划分比例之和必须等于1")
            
        data = list(zip(texts, labels))
        
        if shuffle:
            np.random.seed(42)
            indices = np.random.permutation(len(data))
            data = [data[i] for i in indices]
        
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        self.train_data = data[:n_train]
        self.val_data = data[n_train:n_train + n_val]
        self.test_data = data[n_train + n_val:]
        
        print(f"数据集划分完成:")
        print(f"训练集: {len(self.train_data)} 样本")
        print(f"验证集: {len(self.val_data)} 样本")
        print(f"测试集: {len(self.test_data)} 样本")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_train_data(self) -> Tuple[List[str], List[int]]:
        texts, labels = zip(*self.train_data) if self.train_data else ([], [])
        return list(texts), list(labels)
    
    def get_val_data(self) -> Tuple[List[str], List[int]]:
        texts, labels = zip(*self.val_data) if self.val_data else ([], [])
        return list(texts), list(labels)
    
    def get_test_data(self) -> Tuple[List[str], List[int]]:
        texts, labels = zip(*self.test_data) if self.test_data else ([], [])
        return list(texts), list(labels)
    
    def load_and_prepare_data(self, use_sample_data: bool = True):
        if use_sample_data:
            print("使用示例数据集...")
            texts, labels = self.create_sample_data()
            processed_texts = [self.preprocess_text(text) for text in texts]
            return self.split_data(processed_texts, labels)
        else:
            try:
                # Try to load the sentiment dataset
                if os.path.exists(os.path.join(self.data_dir, "train.tsv", "train.tsv")):
                    print("使用情感分析数据集...")
                    train_texts, train_labels, test_texts, test_labels = self.load_sentiment_dataset()

                    # Process the texts
                    processed_train_texts = [self.preprocess_text(text) for text in train_texts]

                    # Since test data has no labels, we need to split the training data into train/val/test
                    # Convert sentiment labels to binary (0-1 negative, 2-4 positive)
                    binary_labels = [1 if label >= 2 else 0 for label in train_labels]

                    # Split the training data into train/val/test (70%/15%/15%)
                    return self.split_data(processed_train_texts, binary_labels,
                                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
                else:
                    # Fall back to Rotten Tomatoes dataset
                    self.download_rotten_tomatoes_data()
                    texts, labels = self.load_movie_reviews()
                    print(f"成功加载 {len(texts)} 条电影评论")
                    processed_texts = [self.preprocess_text(text) for text in texts]
                    return self.split_data(processed_texts, labels)
            except Exception as e:
                print(f"加载真实数据集失败: {e}")
                print("使用示例数据集...")
                texts, labels = self.create_sample_data()
                processed_texts = [self.preprocess_text(text) for text in texts]
                return self.split_data(processed_texts, labels)

if __name__ == "__main__":
    loader = DataLoader()
    train_data, val_data, test_data = loader.load_and_prepare_data(use_sample_data=False)
    
    train_texts, train_labels = loader.get_train_data()
    print(f"\n训练集示例:")
    for i in range(min(3, len(train_texts))):
        print(f"文本: {train_texts[i]}")
        print(f"标签: {train_labels[i]}")
        print()
