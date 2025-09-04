#!/usr/bin/env python3
"""
简化测试脚本 - 只测试随机嵌入，避免GloVe下载问题
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data.data_loader import DataLoader
from data.dataset import TextDataset, create_data_loaders
from models.cnn_model import CNNTextClassifier
from models.rnn_model import RNNTextClassifier
from training.trainer import Trainer
from utils.text_processor import TextProcessor
from utils.vocabulary import Vocabulary
from utils.metrics import MetricsCalculator

def test_single_model(model_type='cnn', epochs=3):
    """测试单个模型"""
    print(f"测试 {model_type.upper()} 模型 ({epochs} epochs)")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 准备数据
        print("1. 准备数据...")
        text_processor = TextProcessor()
        data_loader = DataLoader(
            data_dir="../data",
            train_file="train.tsv/train.tsv",
            test_file="test.tsv/test.tsv",
            text_processor=text_processor
        )
        
        # 加载数据
        texts, labels, test_texts, test_labels = data_loader.load_data()
        
        # 限制数据量以加快测试
        max_samples = 10000
        if len(texts) > max_samples:
            texts = texts[:max_samples]
            labels = labels[:max_samples]
        
        print(f"训练数据: {len(texts)} 样本")
        
        # 分割数据
        train_texts, train_labels, val_texts, val_labels = data_loader.split_data(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 预处理
        train_texts = data_loader.preprocess_texts(train_texts)
        val_texts = data_loader.preprocess_texts(val_texts)
        
        # 构建词汇表
        vocabulary = Vocabulary(max_size=10000, min_freq=2)
        vocabulary.build_from_texts(train_texts)
        print(f"词汇表大小: {len(vocabulary)}")
        
        # 转换为序列
        train_sequences = [vocabulary.encode(text) for text in train_texts]
        val_sequences = [vocabulary.encode(text) for text in val_texts]
        
        # 填充序列
        max_length = 100  # 减少序列长度以加快训练
        train_sequences = data_loader.pad_sequences(train_sequences, max_length)
        val_sequences = data_loader.pad_sequences(val_sequences, max_length)
        
        # 创建数据集和数据加载器
        train_dataset = TextDataset(train_sequences, train_labels)
        val_dataset = TextDataset(val_sequences, val_labels)
        
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=64, eval_batch_size=128
        )
        
        # 2. 创建模型
        print("2. 创建模型...")
        if model_type == 'cnn':
            model = CNNTextClassifier(
                vocab_size=len(vocabulary),
                embedding_dim=100,  # 减少嵌入维度
                num_classes=2,
                num_filters=64,     # 减少过滤器数量
                filter_sizes=[2, 3, 4],
                dropout=0.5
            )
        else:  # rnn
            model = RNNTextClassifier(
                vocab_size=len(vocabulary),
                embedding_dim=100,  # 减少嵌入维度
                hidden_dim=64,      # 减少隐藏层维度
                num_classes=2,
                num_layers=1,
                dropout=0.5,
                rnn_type='LSTM'
            )
        
        model.print_model_info()
        
        # 3. 训练模型
        print("3. 开始训练...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,  # 使用验证集作为测试集
            device=device,
            config={
                'epochs': epochs,
                'learning_rate': 0.001,
                'optimizer': 'Adam',
                'scheduler': None,
                'early_stopping_patience': 3,
                'gradient_clip': 1.0,
                'save_best_model': False,  # 不保存模型
                'log_interval': 50
            }
        )
        
        # 训练
        training_result = trainer.train()
        
        # 4. 评估
        print("4. 评估模型...")
        test_metrics = trainer.evaluate()
        
        print(f"\n{model_type.upper()} 模型测试结果:")
        print(f"准确率: {test_metrics['accuracy']:.4f}")
        print(f"F1分数: {test_metrics['f1_score']:.4f}")
        print(f"训练时间: {training_result['total_time']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("深度学习文本分类 - 简化测试")
    print("=" * 60)
    
    # 测试CNN模型
    success_cnn = test_single_model('cnn', epochs=2)
    
    print("\n" + "=" * 60)
    
    # 测试RNN模型
    success_rnn = test_single_model('rnn', epochs=2)
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"CNN模型: {'✓ 成功' if success_cnn else '✗ 失败'}")
    print(f"RNN模型: {'✓ 成功' if success_rnn else '✗ 失败'}")
    
    if success_cnn or success_rnn:
        print("\n🎉 至少一个模型测试成功！")
        print("系统基本功能正常，可以进行完整训练:")
        print("python main.py --model cnn --embedding random --epochs 10")
    else:
        print("\n⚠️ 所有模型测试失败，请检查环境配置")

if __name__ == "__main__":
    main()
