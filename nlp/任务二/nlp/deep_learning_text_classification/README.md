# 基于深度学习的文本分类系统

本项目使用PyTorch实现了基于CNN和RNN的文本分类系统，支持多种词嵌入初始化方式。

## 项目结构

```
deep_learning_text_classification/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── main.py                     # 主程序入口
├── config.py                   # 配置文件
├── data/                       # 数据目录
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载器
│   └── dataset.py              # 数据集类
├── models/                     # 模型目录
│   ├── __init__.py
│   ├── cnn_model.py            # CNN文本分类模型
│   ├── rnn_model.py            # RNN文本分类模型
│   └── base_model.py           # 基础模型类
├── embeddings/                 # 词嵌入目录
│   ├── __init__.py
│   ├── embedding_loader.py     # 词嵌入加载器
│   └── glove/                  # GloVe预训练词向量存放目录
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── text_processor.py       # 文本预处理
│   ├── vocabulary.py           # 词汇表构建
│   └── metrics.py              # 评估指标
├── training/                   # 训练相关
│   ├── __init__.py
│   ├── trainer.py              # 训练器
│   └── evaluator.py            # 评估器
└── experiments/                # 实验脚本
    ├── __init__.py
    ├── run_experiments.py      # 实验运行脚本
    └── compare_models.py       # 模型对比脚本
```

## 功能特性

### 1. 深度学习模型
- **CNN模型**: 基于卷积神经网络的文本分类
  - 多种卷积核大小 (3, 4, 5)
  - 最大池化层
  - Dropout正则化
- **RNN模型**: 基于循环神经网络的文本分类
  - LSTM/GRU支持
  - 双向RNN选项
  - 注意力机制 (可选)

### 2. 词嵌入支持
- **随机初始化**: 随机初始化词向量
- **GloVe预训练**: 使用GloVe预训练词向量初始化
- **可训练嵌入**: 支持在训练过程中微调词向量

### 3. 数据处理
- 文本预处理和清洗
- 词汇表构建和管理
- 序列填充和截断
- 数据集划分

### 4. 训练和评估
- 灵活的训练配置
- 多种优化器支持 (Adam, SGD, AdamW)
- 学习率调度
- 早停机制
- 详细的评估指标

## 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境 (可选)
conda create -n text_classification python=3.9.18
conda activate text_classification

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
将数据文件放置在 `../data/` 目录下:
- `train.tsv/train.tsv`: 训练数据
- `test.tsv/test.tsv`: 测试数据

### 3. 下载GloVe词向量 (可选)
```bash
# 下载GloVe 6B词向量
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings/glove/
```

### 4. 运行实验

#### 单个模型训练
```bash
# CNN模型 + 随机初始化
python main.py --model cnn --embedding random

# RNN模型 + GloVe初始化
python main.py --model rnn --embedding glove

# 指定更多参数
python main.py --model cnn --embedding glove --epochs 20 --batch_size 64 --lr 0.001
```

#### 模型对比实验
```bash
# 运行所有模型对比
python main.py --mode compare

# 运行嵌入方式对比
python main.py --mode embedding_compare

# 运行完整实验
python main.py --mode all
```

## 配置说明

主要配置参数在 `config.py` 中定义:

- `MAX_VOCAB_SIZE`: 最大词汇表大小
- `MAX_SEQ_LENGTH`: 最大序列长度
- `EMBEDDING_DIM`: 词嵌入维度
- `HIDDEN_DIM`: 隐藏层维度
- `NUM_CLASSES`: 分类类别数
- `DROPOUT_RATE`: Dropout比率

## 模型架构

### CNN模型
- 词嵌入层 → 多个卷积层 → 最大池化 → 全连接层 → 输出层

### RNN模型
- 词嵌入层 → LSTM/GRU层 → 全连接层 → 输出层

## 实验结果

实验结果将保存在 `results/` 目录下，包括:
- 训练日志
- 模型权重
- 评估报告
- 可视化图表

## 依赖包

- torch >= 1.9.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0

## 许可证

本项目采用MIT许可证。
