#!/usr/bin/env python3

import numpy as np
from data_loader import DataLoader
from feature_extractor import CombinedFeatureExtractor
from classifier import LogisticRegression

def quick_test():
    print("快速测试修复后的代码...")
    
    # 测试数据加载
    print("1. 测试数据加载...")
    loader = DataLoader()
    train_data, val_data, test_data = loader.load_and_prepare_data(use_sample_data=False)
    
    train_texts, train_labels = loader.get_train_data()
    val_texts, val_labels = loader.get_val_data()
    test_texts, test_labels = loader.get_test_data()
    
    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本")
    
    # 检查标签分布
    print(f"训练集标签分布: {np.bincount(train_labels)}")
    print(f"验证集标签分布: {np.bincount(val_labels)}")
    print(f"测试集标签分布: {np.bincount(test_labels)}")
    
    # 测试特征提取
    print("\n2. 测试特征提取...")
    feature_extractor = CombinedFeatureExtractor(
        use_bow=True, 
        use_bigram=True,
        max_features_bow=100,
        max_features_ngram=100
    )
    
    X_train = feature_extractor.fit_transform(train_texts[:1000])  # 使用小样本测试
    X_val = feature_extractor.transform(val_texts[:200])
    X_test = feature_extractor.transform(test_texts[:200])
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"验证特征形状: {X_val.shape}")
    print(f"测试特征形状: {X_test.shape}")
    
    # 测试不同损失函数
    print("\n3. 测试不同损失函数...")
    
    y_train = np.array(train_labels[:1000])
    y_val = np.array(val_labels[:200])
    y_test = np.array(test_labels[:200])
    
    loss_functions = ['cross_entropy', 'squared_error']
    
    for loss_func in loss_functions:
        print(f"\n测试 {loss_func} 损失函数...")
        
        try:
            model = LogisticRegression(
                learning_rate=0.01,
                max_iterations=50,
                loss_function=loss_func
            )
            
            model.fit(X_train, y_train, verbose=False)
            
            train_acc = model.compute_accuracy(y_train, model.predict_proba(X_train))
            val_acc = model.compute_accuracy(y_val, model.predict_proba(X_val))
            test_acc = model.compute_accuracy(y_test, model.predict_proba(X_test))
            
            print(f"  训练准确率: {train_acc:.4f}")
            print(f"  验证准确率: {val_acc:.4f}")
            print(f"  测试准确率: {test_acc:.4f}")
            
        except Exception as e:
            print(f"  错误: {e}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    quick_test()
