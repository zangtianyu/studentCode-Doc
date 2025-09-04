#!/usr/bin/env python3
"""
系统测试脚本 - 验证深度学习文本分类系统是否正常工作
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data.data_loader import DataLoader
from data.dataset import create_data_loaders
from utils.text_processor import TextProcessor
from utils.vocabulary import Vocabulary
from embeddings.embedding_loader import create_embedding_from_config
from models.cnn_model import CNNTextClassifier
from models.rnn_model import RNNTextClassifier
from training.trainer import Trainer

def test_data_loading():
    """测试数据加载"""
    print("1. 测试数据加载...")
    
    try:
        # 创建文本处理器
        text_processor = TextProcessor()
        
        # 创建数据加载器
        data_loader = DataLoader(
            data_dir=DATA_DIR,
            train_file=TRAIN_FILE,
            test_file=TEST_FILE,
            text_processor=text_processor
        )
        
        # 加载数据
        texts, labels, _, _ = data_loader.load_data()
        
        print(f"   成功加载 {len(texts)} 条文本数据")
        print(f"   标签分布: {np.bincount(labels)}")
        
        # 测试文本预处理
        sample_text = texts[0] if texts else "This is a test sentence."
        processed = text_processor.process_text(sample_text)
        print(f"   原始文本: {sample_text[:50]}...")
        print(f"   处理后: {processed[:10]}")
        
        return True
        
    except Exception as e:
        print(f"   数据加载测试失败: {e}")
        return False

def test_vocabulary():
    """测试词汇表构建"""
    print("2. 测试词汇表构建...")
    
    try:
        # 创建测试数据
        test_texts = [
            ["this", "is", "a", "test"],
            ["another", "test", "sentence"],
            ["test", "vocabulary", "building"]
        ]
        
        # 构建词汇表
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build_from_texts(test_texts)
        
        print(f"   词汇表大小: {len(vocab)}")
        print(f"   测试编码: {vocab.encode(['this', 'is', 'unknown'])}")
        
        return True
        
    except Exception as e:
        print(f"   词汇表测试失败: {e}")
        return False

def test_embedding():
    """测试词嵌入"""
    print("3. 测试词嵌入...")
    
    try:
        # 创建简单词汇表
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build_from_texts([["test", "word", "embedding"]])
        
        # 测试随机嵌入
        embedding_config = EMBEDDING_CONFIGS['random']
        embedding_layer = create_embedding_from_config(
            vocabulary=vocab,
            config=embedding_config,
            embedding_dim=50  # 使用较小的维度进行测试
        )
        
        print(f"   嵌入层形状: {embedding_layer.weight.shape}")
        print(f"   嵌入层可训练: {embedding_layer.weight.requires_grad}")
        
        # 测试前向传播
        test_input = torch.tensor([[1, 2, 0]])  # 批次大小1，序列长度3
        output = embedding_layer(test_input)
        print(f"   输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   词嵌入测试失败: {e}")
        return False

def test_models():
    """测试模型创建和前向传播"""
    print("4. 测试模型...")
    
    try:
        # 模型参数
        vocab_size = 100
        embedding_dim = 50
        num_classes = 2
        
        # 测试CNN模型
        print("   测试CNN模型...")
        cnn_model = CNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_filters=10,
            filter_sizes=[2, 3],
            dropout=0.1
        )
        
        # 测试前向传播
        test_input = torch.randint(0, vocab_size, (2, 10))  # 批次大小2，序列长度10
        cnn_output = cnn_model(test_input)
        print(f"   CNN输出形状: {cnn_output.shape}")
        
        # 测试RNN模型
        print("   测试RNN模型...")
        rnn_model = RNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dim=32,
            num_layers=1,
            dropout=0.1
        )
        
        rnn_output = rnn_model(test_input)
        print(f"   RNN输出形状: {rnn_output.shape}")
        
        # 打印模型信息
        cnn_model.print_model_info()
        
        return True
        
    except Exception as e:
        print(f"   模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """测试训练设置"""
    print("5. 测试训练设置...")
    
    try:
        # 创建简单的测试数据
        batch_size = 4
        seq_len = 8
        vocab_size = 50
        num_classes = 2
        
        # 创建模型
        model = CNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=32,
            num_classes=num_classes,
            num_filters=8,
            filter_sizes=[2, 3],
            dropout=0.1
        )
        
        # 创建测试数据
        test_sequences = [
            [1, 2, 3, 4, 5, 0, 0, 0],
            [2, 3, 4, 5, 6, 7, 0, 0],
            [1, 3, 5, 7, 9, 11, 13, 0],
            [2, 4, 6, 8, 10, 12, 14, 16]
        ]
        test_labels = [0, 1, 0, 1]
        
        # 创建数据加载器
        from data.dataset import TextClassificationDataset
        from torch.utils.data import DataLoader
        
        dataset = TextClassificationDataset(test_sequences, test_labels)
        data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # 测试一个训练步骤
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            print(f"   训练步骤成功，损失: {loss.item():.4f}")
            break  # 只测试一个批次
        
        return True
        
    except Exception as e:
        print(f"   训练设置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """测试完整流程（使用小数据集）"""
    print("6. 测试完整流程...")
    
    try:
        # 检查数据是否存在
        train_path = os.path.join(DATA_DIR, TRAIN_FILE)
        if not os.path.exists(train_path):
            print(f"   数据文件不存在: {train_path}")
            print("   跳过完整流程测试")
            return True
        
        # 创建实验运行器
        from experiments.run_experiments import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # 准备数据（只使用小部分数据进行测试）
        print("   准备数据...")
        runner.prepare_data()
        
        # 创建小模型进行快速测试
        embedding_config = EMBEDDING_CONFIGS['random']
        model = runner.create_model('cnn', embedding_config, {
            'num_filters': 8,
            'filter_sizes': [2, 3],
            'dropout': 0.1
        })
        
        print("   模型创建成功")
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试训练配置
        test_config = TRAINING_CONFIG.copy()
        test_config.update({
            'epochs': 1,  # 只训练1个epoch进行测试
            'batch_size': 16,
            'learning_rate': 0.001
        })
        
        print("   开始测试训练...")
        result = runner.train_single_model(model, "测试模型", test_config)
        
        print(f"   测试训练完成!")
        print(f"   验证准确率: {result['training_result']['best_val_accuracy']:.4f}")
        print(f"   测试准确率: {result['test_metrics']['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("深度学习文本分类系统测试")
    print("=" * 40)
    
    # 创建必要的目录
    create_directories()
    
    # 运行各项测试
    tests = [
        test_data_loading,
        test_vocabulary,
        test_embedding,
        test_models,
        test_training_setup,
        test_full_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("   ✓ 通过\n")
            else:
                print("   ✗ 失败\n")
        except Exception as e:
            print(f"   ✗ 异常: {e}\n")
    
    print("=" * 40)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常。")
        print("\n可以开始使用以下命令:")
        print("python main.py --model cnn --embedding random --epochs 5")
    else:
        print("⚠️  部分测试失败，请检查系统配置。")
    
    return passed == total

if __name__ == "__main__":
    main()
