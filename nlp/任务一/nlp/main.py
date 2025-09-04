#!/usr/bin/env python3
import numpy as np
import argparse
import sys
from data_loader import DataLoader
from feature_extractor import CombinedFeatureExtractor, FeatureSelector
from classifier import LogisticRegression, SoftmaxRegression
from trainer import ModelTrainer, ExperimentRunner
from experiments import ExperimentAnalyzer, run_all_experiments

def run_single_experiment():
    
    print("运行单个文本分类实验")
    print("=" * 50)
    
    trainer = ModelTrainer()
    
    print("1. 准备数据...")
    trainer.prepare_data(use_sample_data=False)
    
    print("2. 提取特征...")
    feature_config = {
        'use_bow': True,
        'use_bigram': True,
        'use_trigram': False,
        'max_features_bow': 1000,
        'max_features_ngram': 500
    }
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.extract_features(feature_config)
    
    print("3. 训练模型...")
    model_config = {
        'type': 'logistic',
        'learning_rate': 0.01,
        'max_iterations': 1000,
        'regularization': None
    }
    
    training_config = {
        'batch_strategy': 'full',
        'batch_size': 32,
        'shuffle': True,
        'verbose': True
    }
    
    training_history = trainer.train_model(X_train, y_train, X_val, y_val, model_config, training_config)
    
    print("4. 评估模型...")
    test_evaluation = trainer.get_detailed_evaluation(X_test, y_test)
    
    print(f"\n最终结果:")
    print(f"测试集准确率: {test_evaluation['accuracy']:.4f}")
    if 'f1_score' in test_evaluation:
        print(f"F1分数: {test_evaluation['f1_score']:.4f}")
        print(f"精确率: {test_evaluation['precision']:.4f}")
        print(f"召回率: {test_evaluation['recall']:.4f}")
    
    return trainer, test_evaluation

def run_feature_comparison():
    
    print("运行特征比较实验")
    print("=" * 50)
    
    analyzer = ExperimentAnalyzer()
    results = analyzer.run_feature_comparison_experiments()
    
    analyzer.runner.compare_results()
    analyzer.plot_training_curves(results, "feature_comparison.png")
    
    return results

def run_learning_rate_comparison():
    
    print("运行学习率比较实验")
    print("=" * 50)
    
    analyzer = ExperimentAnalyzer()
    results = analyzer.run_learning_rate_experiments()
    
    analyzer.runner.compare_results()
    analyzer.plot_training_curves(results, "learning_rate_comparison.png")
    
    return results

def run_batch_comparison():

    print("运行批处理策略比较实验")
    print("=" * 50)

    analyzer = ExperimentAnalyzer()
    results = analyzer.run_batch_strategy_experiments()

    analyzer.runner.compare_results()
    analyzer.plot_training_curves(results, "batch_strategy_comparison.png")

    return results

def run_loss_function_comparison():

    print("运行损失函数比较实验")
    print("=" * 50)

    analyzer = ExperimentAnalyzer()
    results = analyzer.run_loss_function_experiments()

    analyzer.runner.compare_results()
    analyzer.plot_training_curves(results, "loss_function_comparison.png")

    return results

def interactive_demo():
    
    print("交互式文本分类演示")
    print("=" * 50)
    
    trainer = ModelTrainer()
    trainer.prepare_data(use_sample_data=False)
    
    feature_config = {
        'use_bow': True,
        'use_bigram': True,
        'max_features_bow': 500,
        'max_features_ngram': 300
    }
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.extract_features(feature_config)
    
    model_config = {
        'type': 'logistic',
        'learning_rate': 0.01,
        'max_iterations': 500
    }
    
    training_config = {
        'batch_strategy': 'full',
        'verbose': False
    }
    
    trainer.train_model(X_train, y_train, X_val, y_val, model_config, training_config)
    
    print(f"模型训练完成！验证集准确率: {trainer.evaluate_model(X_val, y_val):.4f}")
    print("\n现在您可以输入文本进行情感分类预测:")
    print("(输入 'quit' 退出)")
    
    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        processed_text = trainer.data_loader.preprocess_text(text)
        features = trainer.feature_extractor.transform([processed_text])
        
        prediction = trainer.classifier.predict(features)[0]
        probability = trainer.classifier.predict_proba(features)[0]
        
        sentiment = "正面" if prediction == 1 else "负面"
        confidence = probability if prediction == 1 else 1 - probability
        
        print(f"预测结果: {sentiment} (置信度: {confidence:.4f})")

def main():
    parser = argparse.ArgumentParser(description='基于机器学习的文本分类系统')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'feature', 'learning_rate', 'batch', 'loss', 'all', 'demo'],
                       help='运行模式')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            run_single_experiment()
        elif args.mode == 'feature':
            run_feature_comparison()
        elif args.mode == 'learning_rate':
            run_learning_rate_comparison()
        elif args.mode == 'batch':
            run_batch_comparison()
        elif args.mode == 'loss':
            run_loss_function_comparison()
        elif args.mode == 'all':
            run_all_experiments()
        elif args.mode == 'demo':
            interactive_demo()
        else:
            print(f"未知模式: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
