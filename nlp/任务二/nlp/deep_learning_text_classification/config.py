"""
配置文件 - 包含所有模型和训练的超参数配置
"""

import torch
import os

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据配置
DATA_DIR = "../data"
TRAIN_FILE = "train.tsv/train.tsv"
TEST_FILE = "test.tsv/test.tsv"

# 文本处理配置
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 256
MIN_WORD_FREQ = 2
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

# 词嵌入配置
EMBEDDING_DIM = 300
GLOVE_PATH = "embeddings/glove"
GLOVE_FILE = "glove.6B.300d.txt"

# 模型配置
NUM_CLASSES = 2
HIDDEN_DIM = 128
DROPOUT_RATE = 0.5

# CNN特定配置
CNN_CONFIG = {
    'num_filters': 100,
    'filter_sizes': [3, 4, 5],
    'dropout': DROPOUT_RATE
}

# RNN特定配置
RNN_CONFIG = {
    'rnn_type': 'LSTM',  # 'LSTM' or 'GRU'
    'hidden_dim': HIDDEN_DIM,
    'num_layers': 2,
    'bidirectional': True,
    'dropout': DROPOUT_RATE
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'Adam',  # 'Adam', 'SGD', 'AdamW'
    'scheduler': 'StepLR',  # 'StepLR', 'ReduceLROnPlateau', None
    'early_stopping_patience': 5,
    'gradient_clip': 1.0
}

# 评估配置
EVAL_CONFIG = {
    'eval_batch_size': 128,
    'save_best_model': True,
    'save_model_path': 'results/models',
    'log_interval': 100
}

# 实验配置
EXPERIMENT_CONFIG = {
    'random_seed': 42,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'results_dir': 'results',
    'log_dir': 'results/logs',
    'plot_dir': 'results/plots'
}

# 创建必要的目录
def create_directories():
    """创建必要的目录结构"""
    dirs = [
        EXPERIMENT_CONFIG['results_dir'],
        EXPERIMENT_CONFIG['log_dir'],
        EXPERIMENT_CONFIG['plot_dir'],
        EVAL_CONFIG['save_model_path'],
        'embeddings/glove'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# 模型选择配置
MODEL_CONFIGS = {
    'cnn': {
        'model_class': 'CNNTextClassifier',
        'config': CNN_CONFIG
    },
    'rnn': {
        'model_class': 'RNNTextClassifier', 
        'config': RNN_CONFIG
    }
}

# 嵌入方式配置
EMBEDDING_CONFIGS = {
    'random': {
        'type': 'random',
        'trainable': True,
        'description': '随机初始化词嵌入'
    },
    'glove': {
        'type': 'pretrained',
        'path': os.path.join(GLOVE_PATH, GLOVE_FILE),
        'trainable': True,
        'description': 'GloVe预训练词嵌入'
    },
    'glove_frozen': {
        'type': 'pretrained',
        'path': os.path.join(GLOVE_PATH, GLOVE_FILE),
        'trainable': False,
        'description': 'GloVe预训练词嵌入(冻结)'
    }
}
