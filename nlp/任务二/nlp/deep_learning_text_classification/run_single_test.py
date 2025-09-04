#!/usr/bin/env python3
"""
单模型测试脚本 - 最简化版本
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """运行单个CNN模型测试"""
    print("运行CNN模型测试 (2个epoch)")
    print("=" * 40)
    
    try:
        # 导入必要模块
        from config import *
        from data.data_loader import DataLoader
        from data.dataset import TextDataset, create_data_loaders
        from models.cnn_model import CNNTextClassifier
        from training.trainer import Trainer
        from utils.text_processor import TextProcessor
        from utils.vocabulary import Vocabulary
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 1. 数据准备
        print("1. 准备数据...")
        text_processor = TextProcessor()
        data_loader = DataLoader(
            data_dir="../data",
            train_file="train.tsv/train.tsv",
            test_file="test.tsv/test.tsv",
            text_processor=text_processor
        )
        
        # 加载少量数据进行快速测试
        texts, labels, _, _ = data_loader.load_data()
        
        # 只使用前5000个样本
        texts = texts[:5000]
        labels = labels[:5000]
        
        print(f"使用 {len(texts)} 个样本进行测试")
        
        # 分割数据
        train_texts, train_labels, val_texts, val_labels = data_loader.split_data(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 预处理
        train_texts = data_loader.preprocess_texts(train_texts)
        val_texts = data_loader.preprocess_texts(val_texts)
        
        # 构建词汇表
        vocabulary = Vocabulary(max_size=5000, min_freq=2)
        vocabulary.build_from_texts(train_texts)
        print(f"词汇表大小: {len(vocabulary)}")
        
        # 转换为序列
        train_sequences = [vocabulary.encode(text) for text in train_texts]
        val_sequences = [vocabulary.encode(text) for text in val_texts]
        
        # 填充序列
        max_length = 50  # 短序列
        train_sequences = data_loader.pad_sequences(train_sequences, max_length)
        val_sequences = data_loader.pad_sequences(val_sequences, max_length)
        
        # 创建数据集
        train_dataset = TextDataset(train_sequences, train_labels)
        val_dataset = TextDataset(val_sequences, val_labels)
        
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=32, eval_batch_size=64
        )
        
        # 2. 创建小模型
        print("2. 创建CNN模型...")
        model = CNNTextClassifier(
            vocab_size=len(vocabulary),
            embedding_dim=50,   # 小嵌入维度
            num_classes=2,
            num_filters=32,     # 少量过滤器
            filter_sizes=[2, 3],
            dropout=0.3
        )
        
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 3. 训练
        print("3. 开始训练...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            device=device,
            config={
                'epochs': 2,
                'learning_rate': 0.001,
                'optimizer': 'Adam',
                'early_stopping_patience': 10,
                'gradient_clip': 1.0,
                'save_best_model': False,
                'log_interval': 20
            }
        )
        
        # 训练模型
        result = trainer.train()
        
        # 4. 评估
        print("4. 评估模型...")
        metrics = trainer.evaluate()
        
        print(f"\n测试结果:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        print(f"训练时间: {result['total_time']:.2f}秒")
        
        print("\n✅ 测试成功！系统运行正常")
        print("可以运行完整训练:")
        print("python main.py --model cnn --embedding random --epochs 10")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
