#!/usr/bin/env python3

import numpy as np
from data_loader import DataLoader
from feature_extractor import CombinedFeatureExtractor
from classifier import LogisticRegression
from trainer import ModelTrainer

def test_loss_functions():
    print("测试不同损失函数...")
    
    # 创建简单的测试数据
    loader = DataLoader()
    train_data, val_data, test_data = loader.load_and_prepare_data(use_sample_data=True)
    
    train_texts, train_labels = loader.get_train_data()
    val_texts, val_labels = loader.get_val_data()
    test_texts, test_labels = loader.get_test_data()
    
    # 提取特征
    feature_extractor = CombinedFeatureExtractor(
        use_bow=True, 
        use_bigram=True,
        max_features_bow=100,
        max_features_ngram=100
    )
    
    X_train = feature_extractor.fit_transform(train_texts)
    X_val = feature_extractor.transform(val_texts)
    X_test = feature_extractor.transform(test_texts)
    
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)
    
    # 测试不同损失函数
    loss_functions = ['cross_entropy', 'squared_error', 'hinge', 'huber']
    
    results = {}
    
    for loss_func in loss_functions:
        print(f"\n测试 {loss_func} 损失函数...")
        
        try:
            # 创建模型
            model = LogisticRegression(
                learning_rate=0.01,
                max_iterations=100,
                loss_function=loss_func
            )
            
            # 训练模型
            model.fit(X_train, y_train, verbose=False)
            
            # 评估模型
            train_acc = model.compute_accuracy(y_train, model.predict_proba(X_train))
            val_acc = model.compute_accuracy(y_val, model.predict_proba(X_val))
            test_acc = model.compute_accuracy(y_test, model.predict_proba(X_test))
            
            results[loss_func] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'final_cost': model.cost_history[-1] if model.cost_history else 0
            }
            
            print(f"训练准确率: {train_acc:.4f}")
            print(f"验证准确率: {val_acc:.4f}")
            print(f"测试准确率: {test_acc:.4f}")
            print(f"最终损失: {model.cost_history[-1] if model.cost_history else 0:.6f}")
            
        except Exception as e:
            print(f"测试 {loss_func} 时出错: {e}")
            results[loss_func] = {'error': str(e)}
    
    print("\n=" * 50)
    print("损失函数对比结果:")
    print("=" * 50)
    
    for loss_func, result in results.items():
        if 'error' in result:
            print(f"{loss_func}: 错误 - {result['error']}")
        else:
            print(f"{loss_func}:")
            print(f"  测试准确率: {result['test_accuracy']:.4f}")
            print(f"  最终损失: {result['final_cost']:.6f}")
    
    return results

if __name__ == "__main__":
    test_loss_functions()
