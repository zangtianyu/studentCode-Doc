# 基于机器学习的文本分类系统

本项目实现了基于logistic/softmax regression的文本分类系统，使用NumPy从零开始构建，支持多种特征提取方法和训练策略。

## 项目特点

- **纯NumPy实现**: 不依赖sklearn等机器学习库，从零实现所有算法
- **多种特征提取**: 支持Bag-of-Words、N-gram特征提取
- **灵活的训练策略**: 支持full batch、mini-batch、SGD等训练方式
- **完整的实验框架**: 提供系统性的实验比较和分析
- **可视化分析**: 自动生成训练曲线和性能比较图表

## 文件结构

```
├── data_loader.py          # 数据加载和预处理
├── feature_extractor.py    # 特征提取器实现
├── classifier.py           # 分类器核心实现
├── trainer.py             # 训练和评估框架
├── experiments.py         # 实验分析和可视化
├── main.py               # 主程序入口
├── README.md             # 项目说明文档
└── requirements.txt      # 依赖包列表
```

## 核心功能

### 1. 数据处理 (data_loader.py)
- 支持Rotten Tomatoes数据集自动下载
- 提供示例数据集用于快速测试
- 文本预处理和数据集划分功能

### 2. 特征提取 (feature_extractor.py)
- **Bag-of-Words**: 词袋模型特征提取
- **N-gram**: 支持bigram、trigram特征
- **组合特征**: 可组合多种特征类型
- **特征选择**: 基于卡方检验的特征选择

### 3. 分类器 (classifier.py)
- **Logistic回归**: 二分类问题
- **Softmax回归**: 多分类问题
- **正则化**: 支持L1、L2正则化
- **多种优化策略**: 支持不同的批处理方式

### 4. 训练框架 (trainer.py)
- 完整的训练、验证、测试流程
- 支持多种批处理策略
- 详细的性能评估指标

## 安装和使用

### 环境要求
```bash
pip install numpy matplotlib
```

### 快速开始

1. **运行单个实验**:
```bash
python main.py --mode single
```

2. **特征比较实验**:
```bash
python main.py --mode feature
```

3. **学习率比较实验**:
```bash
python main.py --mode learning_rate
```

4. **批处理策略比较**:
```bash
python main.py --mode batch
```

5. **运行所有实验**:
```bash
python main.py --mode all
```

6. **交互式演示**:
```bash
python main.py --mode demo
```

### 详细使用示例

#### 基础使用
```python
from data_loader import DataLoader
from feature_extractor import CombinedFeatureExtractor
from classifier import LogisticRegression

# 加载数据
loader = DataLoader()
train_data, val_data, test_data = loader.load_and_prepare_data()

# 提取特征
extractor = CombinedFeatureExtractor(use_bow=True, use_bigram=True)
train_texts, train_labels = loader.get_train_data()
X_train = extractor.fit_transform(train_texts)

# 训练模型
classifier = LogisticRegression(learning_rate=0.01, max_iterations=1000)
classifier.fit(X_train, np.array(train_labels))

# 预测
predictions = classifier.predict(X_test)
```

#### 实验比较
```python
from experiments import ExperimentAnalyzer

analyzer = ExperimentAnalyzer()

# 运行特征比较实验
feature_results = analyzer.run_feature_comparison_experiments()

# 生成可视化图表
analyzer.plot_training_curves(feature_results, "feature_comparison.png")

# 生成分析报告
analyzer.generate_analysis_report({'特征比较': feature_results})
```

## 实验设计

### 1. 特征比较实验
比较不同特征提取方法的效果：
- BOW特征
- Bigram特征  
- BOW+Bigram组合特征
- 全部特征(BOW+Bigram+Trigram)

### 2. 学习率比较实验
测试不同学习率对模型性能的影响：
- 学习率范围: [0.001, 0.01, 0.1, 0.5, 1.0]

### 3. 批处理策略比较实验
比较不同批处理策略的效果：
- Full Batch: 使用全部训练数据
- Mini-Batch: 使用小批量数据(batch_size=8,16)
- SGD: 随机梯度下降(batch_size=1)

### 4. 正则化比较实验
测试正则化对模型的影响：
- 无正则化
- L1正则化
- L2正则化

## 性能指标

系统提供以下评估指标：
- **准确率(Accuracy)**: 正确预测的样本比例
- **精确率(Precision)**: 正类预测中实际为正类的比例
- **召回率(Recall)**: 实际正类中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **训练时间**: 模型训练所需时间

## 输出文件

运行实验后会生成以下文件：
- `*_curves.png`: 训练曲线图
- `analysis_report.txt`: 详细分析报告
- `experiment_results.json`: 实验结果JSON文件

## 技术实现细节

### 算法实现
- **Sigmoid函数**: 使用数值稳定的实现，避免溢出
- **Softmax函数**: 使用log-sum-exp技巧提高数值稳定性
- **梯度计算**: 基于链式法则的精确梯度计算
- **正则化**: 在损失函数和梯度中正确实现L1/L2正则化

### 特征工程
- **文本预处理**: 小写化、去除标点、空格规范化
- **词汇表构建**: 支持最小/最大文档频率过滤
- **特征向量化**: 高效的稀疏矩阵操作
- **特征选择**: 基于卡方统计量的特征排序

### 训练优化
- **批处理**: 支持多种批处理策略
- **收敛检测**: 基于损失函数变化的早停机制
- **学习率调整**: 固定学习率和自适应调整
- **数据洗牌**: 支持训练数据随机化

## 扩展功能

### 自定义配置
可以通过修改配置参数来调整模型行为：

```python
# 特征配置
feature_config = {
    'use_bow': True,
    'use_bigram': True,
    'use_trigram': False,
    'max_features_bow': 1000,
    'max_features_ngram': 500
}

# 模型配置
model_config = {
    'type': 'logistic',
    'learning_rate': 0.01,
    'max_iterations': 1000,
    'regularization': 'l2',
    'lambda_reg': 0.01
}

# 训练配置
training_config = {
    'batch_strategy': 'mini_batch',
    'batch_size': 32,
    'shuffle': True,
    'verbose': True
}
```

### 添加新的特征提取器
```python
class CustomFeatureExtractor:
    def __init__(self, **kwargs):
        # 初始化参数
        pass
    
    def fit_transform(self, texts):
        # 实现特征提取逻辑
        pass
    
    def transform(self, texts):
        # 实现特征转换逻辑
        pass
```

## 注意事项

1. **数据集**: 默认使用真实的情感分析数据集（位于data目录），包含以下文件：
   - `data/train.tsv/train.tsv`: 训练数据，包含短语和情感标签（0-4）
   - `data/test.tsv/test.tsv`: 测试数据，包含短语但无标签
   - `data/sampleSubmission.csv`: 提交格式示例
2. **内存使用**: 大规模数据集可能需要调整特征数量限制
3. **收敛性**: 某些参数组合可能需要更多迭代次数才能收敛
4. **可视化**: 需要安装matplotlib才能生成图表
5. **依赖**: 需要安装pandas来加载数据集

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：
1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 实验结果分析

### 预期实验结果

基于理论分析，我们预期会观察到以下现象：

1. **特征比较**:
   - BOW+Bigram组合特征通常表现最佳
   - 单独的Bigram特征可能在小数据集上过拟合
   - 添加Trigram特征在小数据集上可能降低性能

2. **学习率影响**:
   - 过小的学习率(0.001)收敛缓慢
   - 过大的学习率(1.0)可能导致震荡或发散
   - 中等学习率(0.01-0.1)通常表现最佳

3. **批处理策略**:
   - Full Batch在小数据集上通常最稳定
   - Mini-Batch提供良好的收敛速度和稳定性平衡
   - SGD收敛速度较慢但可能找到更好的局部最优

4. **正则化效果**:
   - 在小数据集上，正则化有助于防止过拟合
   - L2正则化通常比L1正则化更稳定
   - 正则化强度需要仔细调节

### 性能基准

在示例数据集上的预期性能：
- 基础BOW特征: 准确率 ~0.85
- BOW+Bigram特征: 准确率 ~0.90
- 最优配置: 准确率 >0.90

## 许可证

本项目采用MIT许可证，详见LICENSE文件。
