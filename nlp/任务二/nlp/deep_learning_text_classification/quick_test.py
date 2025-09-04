#!/usr/bin/env python3
"""
快速测试脚本 - 验证系统是否能正常工作
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入"""
    print("1. 测试导入...")
    
    try:
        from config import *
        print("  ✓ 配置导入成功")
    except Exception as e:
        print(f"  ✗ 配置导入失败: {e}")
        return False
    
    try:
        from utils.text_processor import TextProcessor
        print("  ✓ 文本处理器导入成功")
    except Exception as e:
        print(f"  ✗ 文本处理器导入失败: {e}")
        return False
    
    try:
        from utils.vocabulary import Vocabulary
        print("  ✓ 词汇表导入成功")
    except Exception as e:
        print(f"  ✗ 词汇表导入失败: {e}")
        return False
    
    try:
        from models.cnn_model import CNNTextClassifier
        print("  ✓ CNN模型导入成功")
    except Exception as e:
        print(f"  ✗ CNN模型导入失败: {e}")
        return False
    
    return True

def test_text_processing():
    """测试文本处理"""
    print("2. 测试文本处理...")
    
    try:
        from utils.text_processor import TextProcessor
        
        processor = TextProcessor()
        
        # 测试文本处理
        test_text = "This is a test sentence with some words!"
        tokens = processor.process_text(test_text)
        
        print(f"  原始文本: {test_text}")
        print(f"  处理结果: {tokens}")
        print("  ✓ 文本处理成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 文本处理失败: {e}")
        return False

def test_vocabulary():
    """测试词汇表"""
    print("3. 测试词汇表...")
    
    try:
        from utils.vocabulary import Vocabulary
        
        # 创建测试数据
        test_texts = [
            ["this", "is", "a", "test"],
            ["another", "test", "sentence"],
            ["vocabulary", "test", "example"]
        ]
        
        # 构建词汇表
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build_from_texts(test_texts)
        
        print(f"  词汇表大小: {len(vocab)}")
        print(f"  测试编码: {vocab.encode(['this', 'is', 'unknown'])}")
        print("  ✓ 词汇表测试成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 词汇表测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("4. 测试模型创建...")
    
    try:
        from models.cnn_model import CNNTextClassifier
        
        # 创建小模型进行测试
        model = CNNTextClassifier(
            vocab_size=100,
            embedding_dim=50,
            num_classes=2,
            num_filters=8,
            filter_sizes=[2, 3],
            dropout=0.1
        )
        
        # 测试前向传播
        test_input = torch.randint(0, 100, (2, 10))  # 批次大小2，序列长度10
        output = model(test_input)
        
        print(f"  模型输出形状: {output.shape}")
        print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        print("  ✓ 模型创建和前向传播成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("5. 测试数据加载...")
    
    try:
        from data.data_loader import DataLoader
        from utils.text_processor import TextProcessor
        
        # 检查数据文件是否存在
        train_path = "../data/train.tsv/train.tsv"
        if not os.path.exists(train_path):
            print(f"  ⚠ 数据文件不存在: {train_path}")
            print("  跳过数据加载测试")
            return True
        
        # 创建数据加载器
        text_processor = TextProcessor()
        data_loader = DataLoader(
            data_dir="../data",
            train_file="train.tsv/train.tsv",
            test_file="test.tsv/test.tsv",
            text_processor=text_processor
        )
        
        # 加载少量数据进行测试
        texts, labels, _, _ = data_loader.load_data()
        
        print(f"  加载数据数量: {len(texts)}")
        print(f"  标签分布: {set(labels)}")
        print("  ✓ 数据加载成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("深度学习文本分类系统 - 快速测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_text_processing,
        test_vocabulary,
        test_model_creation,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("  ✓ 通过\n")
            else:
                print("  ✗ 失败\n")
        except Exception as e:
            print(f"  ✗ 异常: {e}\n")
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed >= 4:  # 至少4个测试通过
        print("🎉 系统基本功能正常！")
        print("\n建议的运行命令:")
        print("# 训练CNN模型 (5个epoch快速测试)")
        print("python main.py --model cnn --embedding random --epochs 5")
        print("\n# 如果上面成功，可以尝试完整训练")
        print("python main.py --model cnn --embedding random --epochs 20")
    else:
        print("⚠️ 系统存在问题，请检查环境配置")
    
    return passed >= 4

if __name__ == "__main__":
    main()
