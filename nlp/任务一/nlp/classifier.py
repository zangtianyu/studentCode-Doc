import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, regularization: Optional[str] = None,
                 lambda_reg: float = 0.01, loss_function: str = 'cross_entropy'):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.loss_function = loss_function

        self.weights = None
        self.bias = None
        self.cost_history = []
        self.accuracy_history = []
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = len(y_true)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        if self.loss_function == 'cross_entropy':
            cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_function == 'squared_error':
            cost = np.mean((y_true - y_pred) ** 2)
        elif self.loss_function == 'hinge':
            z = np.dot(self.weights, self.weights.T) + self.bias if hasattr(self, 'weights') else 0
            margin = y_true * (2 * y_pred - 1)
            cost = np.mean(np.maximum(0, 1 - margin))
        elif self.loss_function == 'huber':
            delta = 1.0
            residual = np.abs(y_true - y_pred)
            cost = np.mean(np.where(residual <= delta,
                                  0.5 * residual ** 2,
                                  delta * (residual - 0.5 * delta)))
        else:
            raise ValueError(f"不支持的损失函数: {self.loss_function}")

        if self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            cost += self.lambda_reg * np.sum(self.weights ** 2)

        return cost
    
    def compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        m = X.shape[0]

        if self.loss_function == 'cross_entropy':
            dw = (1/m) * np.dot(X.T, (y_pred - y_true))
            db = (1/m) * np.sum(y_pred - y_true)
        elif self.loss_function == 'squared_error':
            error = y_pred - y_true
            dw = (2/m) * np.dot(X.T, error)
            db = (2/m) * np.sum(error)
        elif self.loss_function == 'hinge':
            margin = y_true * (2 * y_pred - 1)
            mask = (margin < 1).astype(float)
            dw = -(1/m) * np.dot(X.T, y_true * mask * 2)
            db = -(1/m) * np.sum(y_true * mask)
        elif self.loss_function == 'huber':
            delta = 1.0
            residual = y_pred - y_true
            mask = np.abs(residual) <= delta
            dw = (1/m) * np.dot(X.T, np.where(mask, residual, delta * np.sign(residual)))
            db = (1/m) * np.sum(np.where(mask, residual, delta * np.sign(residual)))
        else:
            raise ValueError(f"不支持的损失函数: {self.loss_function}")

        if self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_reg * self.weights

        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray, batch_strategy: str = 'full', 
            batch_size: int = 32, shuffle: bool = True, verbose: bool = True):
        
        m, n = X.shape
        
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        
        self.cost_history = []
        self.accuracy_history = []
        
        for iteration in range(self.max_iterations):
            if batch_strategy == 'full':
                X_batch, y_batch = X, y
            elif batch_strategy == 'sgd':
                if shuffle:
                    indices = np.random.permutation(m)
                    X_shuffled, y_shuffled = X[indices], y[indices]
                else:
                    X_shuffled, y_shuffled = X, y
                
                idx = iteration % m
                X_batch, y_batch = X_shuffled[idx:idx+1], y_shuffled[idx:idx+1]
            elif batch_strategy == 'mini_batch':
                if shuffle:
                    indices = np.random.permutation(m)
                    X_shuffled, y_shuffled = X[indices], y[indices]
                else:
                    X_shuffled, y_shuffled = X, y
                
                start_idx = (iteration * batch_size) % m
                end_idx = min(start_idx + batch_size, m)
                X_batch, y_batch = X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]
            else:
                raise ValueError(f"不支持的批处理策略: {batch_strategy}")
            
            z = np.dot(X_batch, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            dw, db = self.compute_gradients(X_batch, y_batch, y_pred)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if iteration % 10 == 0:
                z_full = np.dot(X, self.weights) + self.bias
                y_pred_full = self.sigmoid(z_full)
                cost = self.compute_cost(y, y_pred_full)
                accuracy = self.compute_accuracy(y, y_pred_full)
                
                self.cost_history.append(cost)
                self.accuracy_history.append(accuracy)
                
                if verbose and iteration % 100 == 0:
                    print(f"迭代 {iteration}: 损失 = {cost:.6f}, 准确率 = {accuracy:.4f}")
                
                if len(self.cost_history) > 1:
                    if abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                        if verbose:
                            print(f"在第 {iteration} 次迭代时收敛")
                        break
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true)

class SoftmaxRegression:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, regularization: Optional[str] = None, 
                 lambda_reg: float = 0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.cost_history = []
        self.accuracy_history = []
        
    def softmax(self, z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        cost = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        if self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            cost += self.lambda_reg * np.sum(self.weights ** 2)
        
        return cost
    
    def compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = X.shape[0]
        
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        db = (1/m) * np.sum(y_pred - y_true, axis=0)
        
        if self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_reg * self.weights
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray, batch_strategy: str = 'full', 
            batch_size: int = 32, shuffle: bool = True, verbose: bool = True):
        
        m, n = X.shape
        self.n_classes = len(np.unique(y))
        
        self.weights = np.random.normal(0, 0.01, (n, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        
        y_one_hot = self.one_hot_encode(y)
        
        self.cost_history = []
        self.accuracy_history = []
        
        for iteration in range(self.max_iterations):
            if batch_strategy == 'full':
                X_batch, y_batch = X, y_one_hot
            elif batch_strategy == 'sgd':
                if shuffle:
                    indices = np.random.permutation(m)
                    X_shuffled, y_shuffled = X[indices], y_one_hot[indices]
                else:
                    X_shuffled, y_shuffled = X, y_one_hot
                
                idx = iteration % m
                X_batch, y_batch = X_shuffled[idx:idx+1], y_shuffled[idx:idx+1]
            elif batch_strategy == 'mini_batch':
                if shuffle:
                    indices = np.random.permutation(m)
                    X_shuffled, y_shuffled = X[indices], y_one_hot[indices]
                else:
                    X_shuffled, y_shuffled = X, y_one_hot
                
                start_idx = (iteration * batch_size) % m
                end_idx = min(start_idx + batch_size, m)
                X_batch, y_batch = X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]
            else:
                raise ValueError(f"不支持的批处理策略: {batch_strategy}")
            
            z = np.dot(X_batch, self.weights) + self.bias
            y_pred = self.softmax(z)
            
            dw, db = self.compute_gradients(X_batch, y_batch, y_pred)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if iteration % 10 == 0:
                z_full = np.dot(X, self.weights) + self.bias
                y_pred_full = self.softmax(z_full)
                cost = self.compute_cost(y_one_hot, y_pred_full)
                accuracy = self.compute_accuracy(y, y_pred_full)
                
                self.cost_history.append(cost)
                self.accuracy_history.append(accuracy)
                
                if verbose and iteration % 100 == 0:
                    print(f"迭代 {iteration}: 损失 = {cost:.6f}, 准确率 = {accuracy:.4f}")
                
                if len(self.cost_history) > 1:
                    if abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                        if verbose:
                            print(f"在第 {iteration} 次迭代时收敛")
                        break
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)

if __name__ == "__main__":
    np.random.seed(42)
    
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    print("测试Logistic回归:")
    lr = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    lr.fit(X, y, verbose=True)
    
    predictions = lr.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"训练准确率: {accuracy:.4f}")
    
    print("\n测试Softmax回归:")
    y_multi = np.random.randint(0, 3, 100)
    sr = SoftmaxRegression(learning_rate=0.1, max_iterations=1000)
    sr.fit(X, y_multi, verbose=True)
    
    predictions_multi = sr.predict(X)
    accuracy_multi = np.mean(predictions_multi == y_multi)
    print(f"训练准确率: {accuracy_multi:.4f}")
