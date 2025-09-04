import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from trainer import ExperimentRunner
import json
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ExperimentAnalyzer:
    def __init__(self):
        self.runner = ExperimentRunner()
        
    def run_feature_comparison_experiments(self):
        
        print("运行特征比较实验...")
        
        base_config = {
            'data_config': {'use_sample_data': False},
            'model_config': {
                'type': 'logistic',
                'learning_rate': 0.01,
                'max_iterations': 1000
            },
            'training_config': {
                'batch_strategy': 'full',
                'verbose': False
            }
        }
        
        feature_experiments = [
            {
                'name': 'BOW特征',
                **base_config,
                'feature_config': {
                    'use_bow': True,
                    'use_bigram': False,
                    'use_trigram': False,
                    'max_features_bow': 1000
                }
            },
            {
                'name': 'Bigram特征',
                **base_config,
                'feature_config': {
                    'use_bow': False,
                    'use_bigram': True,
                    'use_trigram': False,
                    'max_features_ngram': 1000
                }
            },
            {
                'name': 'BOW+Bigram特征',
                **base_config,
                'feature_config': {
                    'use_bow': True,
                    'use_bigram': True,
                    'use_trigram': False,
                    'max_features_bow': 500,
                    'max_features_ngram': 500
                }
            },
            {
                'name': '全部特征',
                **base_config,
                'feature_config': {
                    'use_bow': True,
                    'use_bigram': True,
                    'use_trigram': True,
                    'max_features_bow': 400,
                    'max_features_ngram': 300
                }
            }
        ]
        
        return self.runner.run_multiple_experiments(feature_experiments)
    
    def run_learning_rate_experiments(self):
        
        print("运行学习率比较实验...")
        
        base_config = {
            'data_config': {'use_sample_data': False},
            'feature_config': {
                'use_bow': True,
                'use_bigram': True,
                'max_features_bow': 500,
                'max_features_ngram': 500
            },
            'training_config': {
                'batch_strategy': 'full',
                'verbose': False
            }
        }
        
        learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
        lr_experiments = []
        
        for lr in learning_rates:
            lr_experiments.append({
                'name': f'学习率_{lr}',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': lr,
                    'max_iterations': 1000
                }
            })
        
        return self.runner.run_multiple_experiments(lr_experiments)
    
    def run_batch_strategy_experiments(self):
        
        print("运行批处理策略比较实验...")
        
        base_config = {
            'data_config': {'use_sample_data': False},
            'feature_config': {
                'use_bow': True,
                'use_bigram': True,
                'max_features_bow': 500,
                'max_features_ngram': 500
            },
            'model_config': {
                'type': 'logistic',
                'learning_rate': 0.01,
                'max_iterations': 1000
            }
        }
        
        batch_experiments = [
            {
                'name': 'Full_Batch',
                **base_config,
                'training_config': {
                    'batch_strategy': 'full',
                    'verbose': False
                }
            },
            {
                'name': 'Mini_Batch_8',
                **base_config,
                'training_config': {
                    'batch_strategy': 'mini_batch',
                    'batch_size': 8,
                    'shuffle': True,
                    'verbose': False
                }
            },
            {
                'name': 'Mini_Batch_16',
                **base_config,
                'training_config': {
                    'batch_strategy': 'mini_batch',
                    'batch_size': 16,
                    'shuffle': True,
                    'verbose': False
                }
            },
            {
                'name': 'SGD',
                **base_config,
                'training_config': {
                    'batch_strategy': 'sgd',
                    'shuffle': True,
                    'verbose': False
                }
            }
        ]
        
        return self.runner.run_multiple_experiments(batch_experiments)
    
    def run_regularization_experiments(self):
        
        print("运行正则化比较实验...")
        
        base_config = {
            'data_config': {'use_sample_data': False},
            'feature_config': {
                'use_bow': True,
                'use_bigram': True,
                'max_features_bow': 500,
                'max_features_ngram': 500
            },
            'training_config': {
                'batch_strategy': 'full',
                'verbose': False
            }
        }
        
        reg_experiments = [
            {
                'name': '无正则化',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': 0.01,
                    'max_iterations': 1000,
                    'regularization': None
                }
            },
            {
                'name': 'L1正则化',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': 0.01,
                    'max_iterations': 1000,
                    'regularization': 'l1',
                    'lambda_reg': 0.01
                }
            },
            {
                'name': 'L2正则化',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': 0.01,
                    'max_iterations': 1000,
                    'regularization': 'l2',
                    'lambda_reg': 0.01
                }
            }
        ]
        
        return self.runner.run_multiple_experiments(reg_experiments)

    def run_loss_function_experiments(self):

        print("运行损失函数比较实验...")

        base_config = {
            'data_config': {'use_sample_data': False},
            'feature_config': {
                'use_bow': True,
                'use_bigram': True,
                'max_features_bow': 500,
                'max_features_ngram': 500
            },
            'training_config': {
                'batch_strategy': 'full',
                'verbose': False
            }
        }

        loss_experiments = [
            {
                'name': '交叉熵损失',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': 0.01,
                    'max_iterations': 1000,
                    'loss_function': 'cross_entropy'
                }
            },
            {
                'name': '平方误差损失',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': 0.01,
                    'max_iterations': 1000,
                    'loss_function': 'squared_error'
                }
            },
            {
                'name': 'Hinge损失',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': 0.01,
                    'max_iterations': 1000,
                    'loss_function': 'hinge'
                }
            },
            {
                'name': 'Huber损失',
                **base_config,
                'model_config': {
                    'type': 'logistic',
                    'learning_rate': 0.01,
                    'max_iterations': 1000,
                    'loss_function': 'huber'
                }
            }
        ]

        return self.runner.run_multiple_experiments(loss_experiments)

    def plot_training_curves(self, results: List[Dict], save_path: str = "training_curves.png"):
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        for result in results:
            history = result['training_history']
            plt.plot(history['cost_history'], label=result['experiment_name'])
        plt.title('训练损失曲线')
        plt.xlabel('迭代次数 (×10)')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for result in results:
            history = result['training_history']
            plt.plot(history['accuracy_history'], label=result['experiment_name'])
        plt.title('训练准确率曲线')
        plt.xlabel('迭代次数 (×10)')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        names = [r['experiment_name'] for r in results]
        test_accs = [r['test_evaluation']['accuracy'] for r in results]
        plt.bar(names, test_accs)
        plt.title('测试集准确率比较')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        plt.subplot(2, 2, 4)
        train_times = [r['training_history']['training_time'] for r in results]
        plt.bar(names, train_times)
        plt.title('训练时间比较')
        plt.ylabel('时间 (秒)')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练曲线图已保存到: {save_path}")
    
    def generate_analysis_report(self, all_results: Dict[str, List], save_path: str = "analysis_report.txt"):
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("文本分类实验分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            for experiment_type, results in all_results.items():
                f.write(f"{experiment_type}\n")
                f.write("-" * 30 + "\n")
                
                best_result = max(results, key=lambda x: x['test_evaluation']['accuracy'])
                
                f.write(f"最佳配置: {best_result['experiment_name']}\n")
                f.write(f"最佳准确率: {best_result['test_evaluation']['accuracy']:.4f}\n")
                
                if 'f1_score' in best_result['test_evaluation']:
                    f.write(f"F1分数: {best_result['test_evaluation']['f1_score']:.4f}\n")
                    f.write(f"精确率: {best_result['test_evaluation']['precision']:.4f}\n")
                    f.write(f"召回率: {best_result['test_evaluation']['recall']:.4f}\n")
                
                f.write(f"训练时间: {best_result['training_history']['training_time']:.2f}秒\n")
                
                f.write("\n详细结果:\n")
                for result in results:
                    name = result['experiment_name']
                    acc = result['test_evaluation']['accuracy']
                    time = result['training_history']['training_time']
                    f.write(f"  {name}: 准确率={acc:.4f}, 时间={time:.2f}s\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"分析报告已保存到: {save_path}")
    
    def save_results_json(self, all_results: Dict[str, List], save_path: str = "experiment_results.json"):
        
        serializable_results = {}
        for exp_type, results in all_results.items():
            serializable_results[exp_type] = []
            for result in results:
                serializable_result = {
                    'experiment_name': result['experiment_name'],
                    'test_accuracy': result['test_evaluation']['accuracy'],
                    'training_time': result['training_history']['training_time'],
                    'config': result['config']
                }
                if 'f1_score' in result['test_evaluation']:
                    serializable_result['f1_score'] = result['test_evaluation']['f1_score']
                    serializable_result['precision'] = result['test_evaluation']['precision']
                    serializable_result['recall'] = result['test_evaluation']['recall']
                
                serializable_results[exp_type].append(serializable_result)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"实验结果已保存到: {save_path}")

def run_all_experiments():
    
    analyzer = ExperimentAnalyzer()
    
    all_results = {}
    
    all_results['特征比较'] = analyzer.run_feature_comparison_experiments()
    analyzer.runner.results = []
    
    all_results['学习率比较'] = analyzer.run_learning_rate_experiments()
    analyzer.runner.results = []
    
    all_results['批处理策略比较'] = analyzer.run_batch_strategy_experiments()
    analyzer.runner.results = []
    
    all_results['正则化比较'] = analyzer.run_regularization_experiments()
    analyzer.runner.results = []

    all_results['损失函数比较'] = analyzer.run_loss_function_experiments()

    for exp_type, results in all_results.items():
        print(f"\n{exp_type}实验结果:")
        analyzer.runner.results = results
        analyzer.runner.compare_results()
        
        analyzer.plot_training_curves(results, f"{exp_type}_curves.png")
    
    analyzer.generate_analysis_report(all_results)
    analyzer.save_results_json(all_results)
    
    print("\n所有实验完成！")
    return all_results

if __name__ == "__main__":
    results = run_all_experiments()
