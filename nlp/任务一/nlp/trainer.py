import numpy as np
from typing import Dict, List, Tuple, Any
import time
from data_loader import DataLoader
from feature_extractor import CombinedFeatureExtractor, FeatureSelector
from classifier import LogisticRegression, SoftmaxRegression

class ModelTrainer:
    def __init__(self):
        self.data_loader = None
        self.feature_extractor = None
        self.feature_selector = None
        self.classifier = None
        self.training_history = {}
        
    def prepare_data(self, use_sample_data: bool = False, train_ratio: float = 0.7,
                    val_ratio: float = 0.15, test_ratio: float = 0.15):

        self.data_loader = DataLoader()
        train_data, val_data, test_data = self.data_loader.load_and_prepare_data(
            use_sample_data=use_sample_data
        )

        return train_data, val_data, test_data
    
    def extract_features(self, feature_config: Dict[str, Any]):
        
        train_texts, train_labels = self.data_loader.get_train_data()
        val_texts, val_labels = self.data_loader.get_val_data()
        test_texts, test_labels = self.data_loader.get_test_data()
        
        self.feature_extractor = CombinedFeatureExtractor(**feature_config)
        
        print("提取训练集特征...")
        X_train = self.feature_extractor.fit_transform(train_texts)
        
        print("提取验证集特征...")
        X_val = self.feature_extractor.transform(val_texts)
        
        print("提取测试集特征...")
        X_test = self.feature_extractor.transform(test_texts)
        
        return (X_train, np.array(train_labels)), (X_val, np.array(val_labels)), (X_test, np.array(test_labels))
    
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, X_test: np.ndarray, 
                       selection_config: Dict[str, Any]):
        
        if selection_config.get('enabled', False):
            print("进行特征选择...")
            self.feature_selector = FeatureSelector(**{k: v for k, v in selection_config.items() if k != 'enabled'})
            
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_val_selected = self.feature_selector.transform(X_val)
            X_test_selected = self.feature_selector.transform(X_test)
            
            return X_train_selected, X_val_selected, X_test_selected
        else:
            return X_train, X_val, X_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_config: Dict[str, Any], training_config: Dict[str, Any]):
        
        model_type = model_config.get('type', 'logistic')
        
        if model_type == 'logistic':
            self.classifier = LogisticRegression(**{k: v for k, v in model_config.items() if k != 'type'})
        elif model_type == 'softmax':
            self.classifier = SoftmaxRegression(**{k: v for k, v in model_config.items() if k != 'type'})
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        print(f"开始训练 {model_type} 模型...")
        start_time = time.time()
        
        self.classifier.fit(X_train, y_train, **training_config)
        
        training_time = time.time() - start_time
        
        train_accuracy = self.evaluate_model(X_train, y_train)
        val_accuracy = self.evaluate_model(X_val, y_val)
        
        self.training_history = {
            'model_type': model_type,
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'cost_history': self.classifier.cost_history,
            'accuracy_history': self.classifier.accuracy_history,
            'model_config': model_config,
            'training_config': training_config
        }
        
        print(f"训练完成，用时 {training_time:.2f} 秒")
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"验证集准确率: {val_accuracy:.4f}")
        
        return self.training_history
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.classifier is None:
            raise ValueError("模型未训练")
        
        predictions = self.classifier.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_detailed_evaluation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if self.classifier is None:
            raise ValueError("模型未训练")
        
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        accuracy = np.mean(predictions == y)
        
        if len(np.unique(y)) == 2:
            tp = np.sum((predictions == 1) & (y == 1))
            tn = np.sum((predictions == 0) & (y == 0))
            fp = np.sum((predictions == 1) & (y == 0))
            fn = np.sum((predictions == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': {
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                },
                'predictions': predictions,
                'probabilities': probabilities
            }
        else:
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'probabilities': probabilities
            }

class ExperimentRunner:
    def __init__(self):
        self.results = []
        
    def run_experiment(self, experiment_config: Dict[str, Any], experiment_name: str = ""):
        
        print(f"\n{'='*50}")
        print(f"运行实验: {experiment_name}")
        print(f"{'='*50}")
        
        trainer = ModelTrainer()
        
        trainer.prepare_data(**experiment_config.get('data_config', {}))
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.extract_features(
            experiment_config.get('feature_config', {})
        )
        
        X_train, X_val, X_test = trainer.select_features(
            X_train, y_train, X_val, X_test,
            experiment_config.get('feature_selection_config', {'enabled': False})
        )
        
        training_history = trainer.train_model(
            X_train, y_train, X_val, y_val,
            experiment_config.get('model_config', {}),
            experiment_config.get('training_config', {})
        )
        
        test_evaluation = trainer.get_detailed_evaluation(X_test, y_test)
        
        result = {
            'experiment_name': experiment_name,
            'config': experiment_config,
            'training_history': training_history,
            'test_evaluation': test_evaluation,
            'data_shapes': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }
        
        self.results.append(result)
        
        print(f"\n实验结果:")
        print(f"测试集准确率: {test_evaluation['accuracy']:.4f}")
        if 'f1_score' in test_evaluation:
            print(f"F1分数: {test_evaluation['f1_score']:.4f}")
            print(f"精确率: {test_evaluation['precision']:.4f}")
            print(f"召回率: {test_evaluation['recall']:.4f}")
        
        return result
    
    def run_multiple_experiments(self, experiments: List[Dict[str, Any]]):
        
        for i, exp_config in enumerate(experiments):
            exp_name = exp_config.get('name', f'实验_{i+1}')
            self.run_experiment(exp_config, exp_name)
        
        return self.results
    
    def compare_results(self):
        
        if not self.results:
            print("没有实验结果可比较")
            return
        
        print(f"\n{'='*80}")
        print("实验结果比较")
        print(f"{'='*80}")
        
        print(f"{'实验名称':<20} {'测试准确率':<12} {'训练时间(s)':<12} {'特征数':<10} {'批处理策略':<15}")
        print("-" * 80)
        
        for result in self.results:
            name = result['experiment_name'][:18]
            accuracy = result['test_evaluation']['accuracy']
            train_time = result['training_history']['training_time']
            n_features = result['data_shapes']['train'][1]
            batch_strategy = result['config'].get('training_config', {}).get('batch_strategy', 'full')
            
            print(f"{name:<20} {accuracy:<12.4f} {train_time:<12.2f} {n_features:<10} {batch_strategy:<15}")
        
        best_result = max(self.results, key=lambda x: x['test_evaluation']['accuracy'])
        print(f"\n最佳实验: {best_result['experiment_name']}")
        print(f"最佳准确率: {best_result['test_evaluation']['accuracy']:.4f}")

if __name__ == "__main__":
    
    experiment_config = {
        'data_config': {
            'use_sample_data': True
        },
        'feature_config': {
            'use_bow': True,
            'use_bigram': True,
            'use_trigram': False,
            'max_features_bow': 1000,
            'max_features_ngram': 500
        },
        'feature_selection_config': {
            'enabled': False
        },
        'model_config': {
            'type': 'logistic',
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'regularization': None
        },
        'training_config': {
            'batch_strategy': 'full',
            'batch_size': 32,
            'shuffle': True,
            'verbose': True
        }
    }
    
    runner = ExperimentRunner()
    result = runner.run_experiment(experiment_config, "基础实验")
    
    print("\n实验完成！")
