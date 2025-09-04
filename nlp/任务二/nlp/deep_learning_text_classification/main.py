#!/usr/bin/env python3
"""
基于深度学习的文本分类系统 - 主程序
支持CNN和RNN模型，以及多种词嵌入初始化方式
"""

import argparse
import sys
import os
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from experiments.run_experiments import ExperimentRunner
from training.evaluator import ModelEvaluator

def train_single_model(args):
    """训练单个模型"""
    print(f"训练单个模型: {args.model} + {args.embedding}")
    print("=" * 50)
    
    # 创建实验运行器
    runner = ExperimentRunner()
    
    # 获取配置
    embedding_config = EMBEDDING_CONFIGS.get(args.embedding)
    if not embedding_config:
        print(f"不支持的嵌入类型: {args.embedding}")
        return
    
    # 创建模型
    try:
        model = runner.create_model(args.model, embedding_config)
        model_name = f"{args.model.upper()} + {embedding_config['description']}"
        
        # 自定义训练配置
        training_config = TRAINING_CONFIG.copy()
        if args.epochs:
            training_config['epochs'] = args.epochs
        if args.batch_size:
            training_config['batch_size'] = args.batch_size
        if args.lr:
            training_config['learning_rate'] = args.lr
        
        # 训练模型
        result = runner.train_single_model(model, model_name, training_config)
        
        # 打印结果
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {result['training_result']['best_val_accuracy']:.4f}")
        print(f"测试准确率: {result['test_metrics']['accuracy']:.4f}")
        print(f"F1分数: {result['test_metrics']['f1_score']:.4f}")
        print(f"训练时间: {result['training_result']['total_time']:.2f}秒")
        
        # 保存结果
        runner.save_experiment_results({model_name: result}, "single_model")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

def run_model_comparison():
    """运行模型对比实验"""
    print("运行模型对比实验")
    print("=" * 50)
    
    runner = ExperimentRunner()
    results = runner.run_model_comparison()
    
    if results:
        # 保存结果
        runner.save_experiment_results(results, "model_comparison")
        
        # 使用评估器生成详细报告
        evaluator = ModelEvaluator(runner.device)
        
        # 创建模型字典用于评估器
        models_dict = {}
        for model_name, result in results.items():
            # 这里我们只能使用已有的结果，因为模型对象没有保存
            print(f"{model_name}: 测试准确率 = {result['test_metrics']['accuracy']:.4f}")
        
        print("模型对比实验完成!")
    else:
        print("没有成功训练任何模型")

def run_embedding_comparison():
    """运行嵌入方式对比实验"""
    print("运行嵌入方式对比实验")
    print("=" * 50)
    
    runner = ExperimentRunner()
    results = runner.run_embedding_comparison()
    
    if results:
        # 保存结果
        runner.save_experiment_results(results, "embedding_comparison")
        
        print("嵌入方式对比实验完成!")
    else:
        print("没有成功训练任何模型")

def run_all_experiments():
    """运行所有实验"""
    print("运行所有对比实验")
    print("=" * 60)
    
    # 1. 模型对比实验
    print("\n1. 模型对比实验")
    run_model_comparison()
    
    # 2. 嵌入方式对比实验
    print("\n2. 嵌入方式对比实验")
    run_embedding_comparison()
    
    print("\n所有实验完成!")

def download_glove():
    """下载GloVe词向量"""
    print("下载GloVe词向量...")
    
    from embeddings.embedding_loader import download_and_prepare_glove
    
    try:
        glove_path = download_and_prepare_glove(
            save_dir="embeddings/glove",
            dim=EMBEDDING_DIM
        )
        print(f"GloVe词向量下载完成: {glove_path}")
    except Exception as e:
        print(f"下载失败: {e}")

def interactive_demo():
    """交互式演示"""
    print("交互式文本分类演示")
    print("=" * 30)
    print("功能开发中...")
    # TODO: 实现交互式演示功能

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='基于深度学习的文本分类系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练CNN模型 + 随机嵌入
  python main.py --model cnn --embedding random
  
  # 训练RNN模型 + GloVe嵌入
  python main.py --model rnn --embedding glove --epochs 20 --batch_size 64
  
  # 运行模型对比实验
  python main.py --mode compare
  
  # 运行嵌入方式对比实验
  python main.py --mode embedding_compare
  
  # 运行所有实验
  python main.py --mode all
  
  # 下载GloVe词向量
  python main.py --mode download_glove
        """
    )
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'compare', 'embedding_compare', 'all', 'download_glove', 'demo'],
                       help='运行模式')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'rnn', 'attention_rnn'],
                       help='模型类型')
    
    parser.add_argument('--embedding', type=str, default='random',
                       choices=['random', 'glove', 'glove_frozen'],
                       help='词嵌入类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    
    # 其他参数
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU设备ID')
    
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.gpu is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print(f"使用GPU: {args.gpu}")
        else:
            print("CUDA不可用，使用CPU")
    
    # 设置随机种子
    if args.seed is not None:
        EXPERIMENT_CONFIG['random_seed'] = args.seed
    
    try:
        # 根据模式运行相应功能
        if args.mode == 'single':
            train_single_model(args)
        elif args.mode == 'compare':
            run_model_comparison()
        elif args.mode == 'embedding_compare':
            run_embedding_comparison()
        elif args.mode == 'all':
            run_all_experiments()
        elif args.mode == 'download_glove':
            download_glove()
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
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
