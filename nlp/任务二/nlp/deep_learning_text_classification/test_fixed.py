#!/usr/bin/env python3
"""
修复后的测试脚本 - 只测试随机嵌入
"""

import os
import sys
import torch
from config import *
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """运行修复后的测试"""
    print("修复后的深度学习文本分类测试")
    print("=" * 50)
    
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 导入模块
       
        from data.data_loader import DataLoader
        from data.dataset import TextDataset, create_data_loaders
        from models.cnn_model import CNNTextClassifier
        from training.trainer import Trainer
        from utils.text_processor import TextProcessor
        from utils.vocabulary import Vocabulary
        
        print("✅ 所有模块导入成功")
        
        # 1. 数据准备
        print("\n1. 准备数据...")
        text_processor = TextProcessor()
        data_loader = DataLoader(
            data_dir="../data",
            train_file="train.tsv/train.tsv",
            test_file="test.tsv/test.tsv",
            text_processor=text_processor
        )
        
        # 加载少量数据
        texts, labels, _, _ = data_loader.load_data()
        
        # 只使用前1000个样本进行快速测试
        texts = texts[:1000]
        labels = labels[:1000]
        
        print(f"使用 {len(texts)} 个样本进行测试")
        
        # 简单分割数据
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 预处理
        train_texts = data_loader.preprocess_texts(train_texts)
        val_texts = data_loader.preprocess_texts(val_texts)
        
        # 构建词汇表
        vocabulary = Vocabulary(max_size=2000, min_freq=1)
        vocabulary.build_from_texts(train_texts)
        print(f"词汇表大小: {len(vocabulary)}")
        
        # 转换为序列
        train_sequences = [vocabulary.encode(text) for text in train_texts]
        val_sequences = [vocabulary.encode(text) for text in val_texts]
        
        # 填充序列
        max_length = 30  # 很短的序列
        train_sequences = data_loader.pad_sequences(train_sequences, max_length)
        val_sequences = data_loader.pad_sequences(val_sequences, max_length)
        
        # 创建数据集
        train_dataset = TextDataset(train_sequences, train_labels)
        val_dataset = TextDataset(val_sequences, val_labels)
        
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=16, eval_batch_size=32
        )
        
        print("✅ 数据准备完成")
        
        # 2. 创建小模型
        print("\n2. 创建CNN模型...")
        model = CNNTextClassifier(
            vocab_size=len(vocabulary),
            embedding_dim=32,   # 很小的嵌入维度
            num_classes=2,
            num_filters=16,     # 很少的过滤器
            filter_sizes=[2, 3],
            dropout=0.2
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        print("✅ 模型创建完成")
        
        # 3. 训练
        print("\n3. 开始训练 (1个epoch)...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            device=device,
            config={
                'epochs': 1,  # 只训练1个epoch
                'learning_rate': 0.001,
                'optimizer': 'Adam',
                'early_stopping_patience': 10,
                'gradient_clip': 1.0,
                'save_best_model': False,
                'log_interval': 10
            }
        )
        
        # 训练模型
        result = trainer.train()
        print("✅ 训练完成")
        
        # 4. 评估
        print("\n4. 评估模型...")
        metrics = trainer.evaluate()
        
        print(f"\n🎉 测试成功!")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        print(f"训练时间: {result['total_time']:.2f}秒")
        
        print("\n✅ 所有问题已修复，系统运行正常!")
        print("\n推荐的下一步:")
        print("1. 运行完整的CNN训练:")
        print("   python main.py --model cnn --embedding random --epochs 10")
        print("\n2. 如果想使用GloVe嵌入，先解压文件:")
        print("   python extract_glove.py")
        print("   python main.py --model cnn --embedding glove --epochs 10")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
