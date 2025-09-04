

# import torch
# import torch.nn.functional as F
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset, ZINC
# from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
# from torch_geometric.nn import global_add_pool  
# from torch_scatter import scatter
# from torch.nn import Linear, Sequential, ReLU
# import time
# import numpy as np
# from sklearn.metrics import roc_auc_score, average_precision_score
# from torch_geometric.loader import NeighborLoader
# from sklearn.preprocessing import label_binarize
# from sklearn.model_selection import train_test_split  # 新增分层抽样工具
# import warnings
# import torch.serialization
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
# from torch_geometric.data import InMemoryDataset

# # 解决安全加载警告
# torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr])

# # 忽略已处理的警告
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# # 固定随机种子（确保实验可复现）
# torch.manual_seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
#     torch.cuda.manual_seed_all(42)

# # 自定义全局最小池化
# def global_min_pool(x, batch):
#     return scatter(x, batch, dim=0, reduce='min')

# # GCN模型
# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes, pooling, num_layers):
#         super(GCN, self).__init__()
#         hidden = 64
#         self.convs = torch.nn.ModuleList()
#         in_channels = num_features
#         for _ in range(num_layers):
#             out_channels = hidden
#             self.convs.append(GCNConv(in_channels, out_channels))
#             in_channels = out_channels
#         self.fc = Linear(hidden, num_classes)
#         self.pooling = pooling

#     def forward(self, x, edge_index, batch):
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index))
#         x = self.pooling(x, batch)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)

# # GAT模型（调整dropout为0.3，避免过强正则化）
# class GAT(torch.nn.Module):
#     def __init__(self, num_features, num_classes, pooling, num_layers):
#         super(GAT, self).__init__()
#         hidden = 64
#         heads = 8
#         self.convs = torch.nn.ModuleList()
#         in_channels = num_features
#         for i in range(num_layers):
#             out_channels = hidden // heads if i < num_layers - 1 else hidden
#             concat = True if i < num_layers - 1 else False
#             self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=0.3))  # 原0.6→0.3
#             in_channels = hidden if concat else out_channels
#         self.fc = Linear(hidden, num_classes)
#         self.pooling = pooling

#     def forward(self, x, edge_index, batch):
#         for conv in self.convs:
#             x = F.elu(conv(x, edge_index))
#         x = self.pooling(x, batch)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)

# # GraphSAGE模型
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, num_features, num_classes, pooling, num_layers):
#         super(GraphSAGE, self).__init__()
#         hidden_dim = 64
#         self.convs = torch.nn.ModuleList()
#         in_dim = num_features
#         for _ in range(num_layers):
#             self.convs.append(SAGEConv(in_dim, hidden_dim))
#             in_dim = hidden_dim
#         self.fc = torch.nn.Linear(hidden_dim, num_classes)
#         self.pooling = pooling

#     def forward(self, x, edge_index, batch):
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index))
#         x = self.pooling(x, batch)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)

# # GIN模型
# class GIN(torch.nn.Module):
#     def __init__(self, num_features, num_classes, pooling, num_layers):
#         super(GIN, self).__init__()
#         dim = 64
#         self.convs = torch.nn.ModuleList()
#         in_dim = num_features
#         for _ in range(num_layers):
#             nn_ = Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim))
#             self.convs.append(GINConv(nn_))
#             in_dim = dim
#         self.fc = Linear(dim, num_classes)
#         self.pooling = pooling

#     def forward(self, x, edge_index, batch):
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index))
#         x = self.pooling(x, batch)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)

# # 训练函数（保持不变）
# def train(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch)
#         loss = F.nll_loss(out, data.y.squeeze())
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)

# # 创建子图加载器（保持不变）
# def create_subgraph_loader(train_dataset, batch_size, num_neighbors, num_subgraphs_per_graph=3):
#     sub_data_list = []
#     for data in train_dataset:
#         if data.num_nodes < sum(num_neighbors):
#             continue
#         loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size)
#         for i, subgraph in enumerate(loader):
#             subgraph.y = data.y.clone()
#             sub_data_list.append(subgraph)
#             if i >= num_subgraphs_per_graph - 1:
#                 break
#     return DataLoader(sub_data_list, batch_size=32, shuffle=True)

# # 修复后的测试函数：处理类别缺失、单类别等异常情况
# def test(model, loader, device, num_classes, all_classes):
#     """
#     all_classes: 数据集完整类别列表（如ENZYMES的[0,1,2,3,4,5]）
#     解决AUC为NaN的核心：仅对测试集中存在且有正样本的类别计算指标
#     """
#     model.eval()
#     y_true = []
#     y_score = []
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             out = model(data.x, data.edge_index, data.batch)
#             pred = out.argmax(dim=1)
            
#             # 确保标签格式统一（转为列表避免维度问题）
#             y_true.extend(data.y.squeeze().cpu().numpy().tolist())
#             y_score.extend(out.exp().cpu().detach().numpy().tolist())
            
#             # 计算准确率
#             correct += int((pred == data.y.squeeze()).sum())
#             total += data.y.size(0)
    
#     # 转换为numpy数组
#     y_true = np.array(y_true)
#     y_score = np.array(y_score)
#     acc = correct / total if total > 0 else 0
    
#     # 情况1：测试集无样本或仅含1个类别→无法计算AUC/AP
#     if len(y_true) == 0 or len(np.unique(y_true)) == 1:
#         return acc, np.nan, np.nan
    
#     # 情况2：测试集含多个类别→筛选有效类别计算指标
#     present_classes = np.unique(y_true)  # 测试集中实际存在的类别
#     # 按完整类别编码标签（确保与模型输出维度一致）
#     y_true_bin = label_binarize(y_true, classes=all_classes)
#     # 仅保留测试集中存在类别的得分和标签（排除缺失类别干扰）
#     y_score_filtered = y_score[:, present_classes]
#     y_true_bin_filtered = y_true_bin[:, present_classes]
    
#     # 计算AUC（宏平均，跳过无正样本的类别）
#     auc = np.nan
#     try:
#         class_aucs = []
#         for i in range(len(present_classes)):
#             # 确保该类别有正样本（避免全0标签）
#             if np.sum(y_true_bin_filtered[:, i]) > 0:
#                 class_auc = roc_auc_score(y_true_bin_filtered[:, i], y_score_filtered[:, i])
#                 class_aucs.append(class_auc)
#         if len(class_aucs) > 0:
#             auc = np.mean(class_aucs)
#     except Exception:
#         auc = np.nan
    
#     # 计算AP（与AUC逻辑一致）
#     ap = np.nan
#     try:
#         class_aps = []
#         for i in range(len(present_classes)):
#             if np.sum(y_true_bin_filtered[:, i]) > 0:
#                 class_ap = average_precision_score(y_true_bin_filtered[:, i], y_score_filtered[:, i])
#                 class_aps.append(class_ap)
#         if len(class_aps) > 0:
#             ap = np.mean(class_aps)
#     except Exception:
#         ap = np.nan
    
#     return acc, auc, ap

# # 自定义数据集加载器（解决torch.load警告，保持不变）
# class SafeTUDataset(TUDataset):
#     def __init__(self, root, name, transform=None, pre_transform=None):
#         super().__init__(root, name, transform=transform, pre_transform=pre_transform)
    
#     def _load(self):
#         data, slices = torch.load(self.processed_paths[0], weights_only=True)
#         return data, slices

# class SafeZINC(ZINC):
#     def __init__(self, root, subset=False, transform=None, pre_transform=None):
#         super().__init__(root, subset=subset, transform=transform, pre_transform=pre_transform)
    
#     def _load(self):
#         data, slices = torch.load(self.processed_paths[0], weights_only=True)
#         return data, slices

# # 主程序（核心修改：分层抽样+传递完整类别列表）
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # 使用自定义的安全数据集加载器
#     datasets = {
#         'ENZYMES': SafeTUDataset(root='./dataset', name='ENZYMES'),
#         'ZINC': SafeZINC(root='./dataset/ZINC', subset=True)
#     }

#     # 池化方法（保持不变）
#     pooling_methods = {
#         'AvgPooling': global_mean_pool,
#         'MaxPooling': global_max_pool,
#         'MinPooling': global_min_pool,
#     }

#     # 模型（保持不变）
#     model_classes = {
#         'GCN': GCN,
#         'GAT': GAT,
#         'GraphSAGE': GraphSAGE,
#         'GIN': GIN,
#     }

#     num_layers_list = [1, 2]
#     lr_list = [0.01, 0.03]
#     modes = ['full', 'sampled']

#     for dataset_name, dataset in datasets.items():
#         print(f"\nProcessing dataset: {dataset_name}")
        
#         # 1. 处理数据集类别（ENZYMES为分类，ZINC为回归转分类）
#         if dataset_name == 'ZINC':
#             ys = [d.y.item() for d in dataset]
#             bins = np.percentile(ys, [25, 50, 75])
            
#             def get_class(v):
#                 if v < bins[0]:
#                     return 0
#                 elif v < bins[1]:
#                     return 1
#                 elif v < bins[2]:
#                     return 2
#                 else:
#                     return 3
            
#             for d in dataset:
#                 d.y = torch.tensor([get_class(d.y.item())])
#             num_classes = 4
#             all_classes = list(range(num_classes))  # ZINC完整类别：[0,1,2,3]
#         else:
#             num_classes = dataset.num_classes
#             all_classes = list(range(num_classes))  # ENZYMES完整类别：[0,1,2,3,4,5]
        
#         # 2. 分层抽样分割数据集（核心修复：确保测试集类别平衡）
#         # 提取所有样本标签用于分层
#         labels = np.array([data.y.item() for data in dataset])
#         # 分层分割（test_size=0.2，按标签分布抽样）
#         train_idx, test_idx = train_test_split(
#             range(len(dataset)),
#             test_size=0.2,
#             stratify=labels,  # 关键参数：保证训练/测试集类别分布与原数据集一致
#             random_state=42
#         )
#         # 按索引获取训练/测试集
#         train_ds = dataset[train_idx]
#         test_ds = dataset[test_idx]
        
#         # 打印数据集信息（验证类别分布）
#         test_labels = [data.y.item() for data in test_ds]
#         test_class_dist = np.bincount(test_labels)  # 测试集类别分布
#         print(f"Dataset size: {len(dataset)}, Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {num_classes}")
#         print(f"Test set class distribution: {test_class_dist}")  # 如ENZYMES应接近[20,20,20,20,20,20]

#         # 3. 遍历所有参数组合训练+测试
#         for pool_name, pool in pooling_methods.items():
#             for model_name, model_class in model_classes.items():
#                 for num_layers in num_layers_list:
#                     for lr in lr_list:
#                         for mode in modes:
#                             try:
#                                 # 初始化模型
#                                 model = model_class(
#                                     dataset.num_node_features, 
#                                     num_classes, 
#                                     pool, 
#                                     num_layers
#                                 ).to(device)
#                                 optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#                                 # 创建数据加载器
#                                 if mode == 'full':
#                                     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
#                                 else:
#                                     train_loader = create_subgraph_loader(
#                                         train_ds, 
#                                         batch_size=16, 
#                                         num_neighbors=[25, 10]
#                                     )
#                                     if len(train_loader.dataset) == 0:
#                                         print(f"Skipping {model_name}-{pool_name}-Layers{num_layers}-LR{lr}-{mode}: No subgraphs")
#                                         continue

#                                 test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

#                                 # 训练模型（增加到100轮，确保收敛）
#                                 start_time = time.time()
#                                 for epoch in range(1, 10):  #
#                                     loss = train(model, train_loader, optimizer, device)
#                                     if epoch % 5 == 0:  # 每20轮打印一次损失，减少输出冗余
#                                         print(f"[{dataset_name}-{model_name}-{pool_name}] Epoch {epoch}, Loss: {loss:.4f}")
#                                 training_time = time.time() - start_time

#                                 # 测试模型（传递完整类别列表all_classes）
#                                 acc, auc, ap = test(model, test_loader, device, num_classes, all_classes)
                                
#                                 # 输出结果（保留4位小数，增加可读性）
#                                 print(f'\nResult - Dataset: {dataset_name}, Model: {model_name}, Pooling: {pool_name}, '
#                                       f'Layers: {num_layers}, LR: {lr}, Mode: {mode}\n'
#                                       f'Accuracy: {acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, Time: {training_time:.2f}s\n')
                                
#                             except Exception as e:
#                                 print(f"\nError with combination - Dataset: {dataset_name}, Model: {model_name}, "
#                                       f"Pooling: {pool_name}, Layers: {num_layers}, LR: {lr}, Mode: {mode}")
#                                 print(f"Error message: {str(e)}\n")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn import global_add_pool  
from torch_scatter import scatter
from torch.nn import Linear, Sequential, ReLU
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import warnings
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data import InMemoryDataset

# 解决安全加载警告
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr])

# 忽略已处理的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# 固定随机种子（确保实验可复现）
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# 自定义全局最小池化
def global_min_pool(x, batch):
    return scatter(x, batch, dim=0, reduce='min')

# 通用模型基类，提取共同功能
class BaseGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, pooling, num_layers):
        super().__init__()
        self.pooling = pooling
        self.num_classes = num_classes
        
    def forward(self, x, edge_index, batch):
        # 确保输入特征为float类型
        x = x.float()
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        x = self.pooling(x, batch)
        x = self.fc(x)
        
        # 对于二分类或多分类使用不同的激活函数
        if self.num_classes == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

# GCN模型
class GCN(BaseGNN):
    def __init__(self, num_features, num_classes, pooling, num_layers):
        super().__init__(num_features, num_classes, pooling, num_layers)
        hidden = 64
        self.activation = F.relu
        self.convs = torch.nn.ModuleList()
        in_channels = num_features
        for _ in range(num_layers):
            out_channels = hidden
            self.convs.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels
        # 确保输出层维度与类别数量匹配
        self.fc = Linear(hidden, num_classes if num_classes > 1 else 1)

# GAT模型
class GAT(BaseGNN):
    def __init__(self, num_features, num_classes, pooling, num_layers):
        super().__init__(num_features, num_classes, pooling, num_layers)
        hidden = 64
        heads = 8
        self.activation = F.elu
        self.convs = torch.nn.ModuleList()
        in_channels = num_features
        for i in range(num_layers):
            out_channels = hidden // heads if i < num_layers - 1 else hidden
            concat = True if i < num_layers - 1 else False
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=0.3))
            in_channels = hidden if concat else out_channels
        # 确保输出层维度与类别数量匹配
        self.fc = Linear(hidden, num_classes if num_classes > 1 else 1)

# GraphSAGE模型
class GraphSAGE(BaseGNN):
    def __init__(self, num_features, num_classes, pooling, num_layers):
        super().__init__(num_features, num_classes, pooling, num_layers)
        hidden_dim = 64
        self.activation = F.relu
        self.convs = torch.nn.ModuleList()
        in_dim = num_features
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        # 确保输出层维度与类别数量匹配
        self.fc = torch.nn.Linear(hidden_dim, num_classes if num_classes > 1 else 1)

# GIN模型
class GIN(BaseGNN):
    def __init__(self, num_features, num_classes, pooling, num_layers):
        super().__init__(num_features, num_classes, pooling, num_layers)
        dim = 64
        self.activation = F.relu
        self.convs = torch.nn.ModuleList()
        in_dim = num_features
        for _ in range(num_layers):
            nn_ = Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim))
            self.convs.append(GINConv(nn_))
            in_dim = dim
        # 确保输出层维度与类别数量匹配
        self.fc = Linear(dim, num_classes if num_classes > 1 else 1)

# 训练函数
def train(model, loader, optimizer, device, num_classes):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        
        # 确保标签在有效范围内
        y = data.y.squeeze().long()
        y = torch.clamp(y, 0, num_classes - 1)
        
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        
        # 根据类别数量使用不同的损失函数
        if num_classes == 1:
            # 二分类使用BCEWithLogitsLoss
            loss = F.binary_cross_entropy(out.squeeze(), y.float())
        else:
            # 多分类使用负对数似然损失
            loss = F.nll_loss(out, y)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 创建子图加载器
def create_subgraph_loader(train_dataset, batch_size, num_neighbors, num_subgraphs_per_graph=3):
    sub_data_list = []
    for data in train_dataset:
        if data.num_nodes < sum(num_neighbors):
            continue
        loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size)
        for i, subgraph in enumerate(loader):
            subgraph.y = data.y.clone()
            sub_data_list.append(subgraph)
            if i >= num_subgraphs_per_graph - 1:
                break
    return DataLoader(sub_data_list, batch_size=32, shuffle=True) if sub_data_list else None

# 测试函数
def test(model, loader, device, num_classes, all_classes):
    model.eval()
    y_true = []
    y_score = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # 确保标签在有效范围内
            y = data.y.squeeze().long()
            y_clamped = torch.clamp(y, 0, num_classes - 1)
            
            out = model(data.x, data.edge_index, data.batch)
            
            # 根据类别数量处理预测结果
            if num_classes == 1:
                # 二分类：使用0.5作为阈值
                pred = (out.squeeze() > 0.5).long()
                # 收集正类的概率
                y_score.extend(out.squeeze().cpu().detach().numpy().tolist())
            else:
                # 多分类：取概率最大的类别
                pred = out.argmax(dim=1)
                # 收集所有类别的概率分布
                y_score.extend(out.exp().cpu().detach().numpy().tolist())
            
            y_true.extend(y_clamped.cpu().numpy().tolist())
            correct += int((pred == y_clamped).sum())
            total += y_clamped.size(0)
    
    y_true = np.array(y_true)
    acc = correct / total if total > 0 else 0
    
    # 处理评估指标计算
    auc = np.nan
    ap = np.nan
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        # 确保类别列表正确
        present_classes = np.unique(y_true)
        n_classes = len(present_classes)
        
        try:
            if num_classes == 1:
                # 二分类情况 - 确保y_true和y_score都是1维的
                if len(y_score) > 0 and y_true.ndim == 1 and np.array(y_score).ndim == 1:
                    auc = roc_auc_score(y_true, y_score)
                    ap = average_precision_score(y_true, y_score)
                else:
                    print(f"Metrics shape error: y_true shape {y_true.shape}, y_score shape {np.array(y_score).shape}")
            elif num_classes == 2:
                # 二分类但使用了2个输出节点的情况
                if len(y_score) > 0 and y_true.ndim == 1:
                    # 提取正类的概率
                    if np.array(y_score).ndim == 2 and np.array(y_score).shape[1] == 2:
                        pos_probs = np.array(y_score)[:, 1]
                        auc = roc_auc_score(y_true, pos_probs)
                        ap = average_precision_score(y_true, pos_probs)
                    else:
                        auc = roc_auc_score(y_true, y_score)
                        ap = average_precision_score(y_true, y_score)
            else:
                # 多分类情况
                y_true_bin = label_binarize(y_true, classes=all_classes)
                if len(y_score[0]) == len(all_classes):
                    auc = roc_auc_score(y_true_bin, y_score, average='macro')
                    ap = average_precision_score(y_true_bin, y_score, average='macro')
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
    
    return acc, auc, ap

# 自定义数据集加载器
class SafeTUDataset(TUDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super().__init__(root, name, transform=transform, pre_transform=pre_transform)
    
    def _load(self):
        data, slices = torch.load(self.processed_paths[0], weights_only=True)
        return data, slices

class SafeZINC(ZINC):
    def __init__(self, root, subset=False, transform=None, pre_transform=None):
        super().__init__(root, subset=subset, transform=transform, pre_transform=pre_transform)
    
    def _load(self):
        data, slices = torch.load(self.processed_paths[0], weights_only=True)
        return data, slices

# 主程序
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据集
    datasets = {
        # 'ENZYMES': SafeTUDataset(root='./dataset', name='ENZYMES'),
        'ZINC': SafeZINC(root='./dataset/ZINC', subset=True)
    }

    # 池化方法
    pooling_methods = {
        'AvgPooling': global_mean_pool,
        'MaxPooling': global_max_pool,
        'MinPooling': global_min_pool,
    }

    # 模型
    model_classes = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'GIN': GIN,
    }

    num_layers_list = [1, 2]
    lr_list = [0.01, 0.03]
    modes = ['full', 'sampled']

    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        # 处理数据集类别
        if dataset_name == 'ZINC':
            ys = [d.y.item() for d in dataset]
            # 调整为2分类以避免样本数不足问题
            median = np.median(ys)
            
            def get_class(v):
                # 确保返回非负整数标签
                return 0 if v < median else 1
            
            # 确保标签是整数类型且非负
            for d in dataset:
                label = get_class(d.y.item())
                # 额外检查确保标签非负且在0-1范围内
                if label < 0:
                    label = 0
                elif label >= 2:
                    label = 1
                d.y = torch.tensor([label], dtype=torch.long)
            num_classes = 2  # 明确设置为二分类
            all_classes = list(range(num_classes))
        else:
            num_classes = dataset.num_classes
            all_classes = list(range(num_classes))
        
        # 提取所有样本标签并转换为整数
        labels = np.array([data.y.item() for data in dataset], dtype=np.int64)
        
        # 确保标签非负且在有效范围内
        if np.any(labels < 0) or np.any(labels >= num_classes):
            print(f"Warning: Labels out of valid range [0, {num_classes-1}], converting to valid range")
            labels = np.clip(labels, 0, num_classes - 1)
        
        # 分割数据集（带错误处理的分层抽样）
        try:
            # 检查每个类别的样本数是否至少为2
            class_counts = np.bincount(labels)
            if len(class_counts) < num_classes:
                # 补充缺失的类别计数
                class_counts = np.pad(class_counts, (0, num_classes - len(class_counts)), mode='constant')
            if np.min(class_counts) < 2:
                raise ValueError("Some classes have fewer than 2 samples")
            
            # 尝试分层抽样
            train_idx, test_idx = train_test_split(
                range(len(dataset)),
                test_size=0.2,
                stratify=labels,
                random_state=42
            )
            print("Using stratified train-test split")
        except ValueError:
            # 分层抽样失败时使用普通随机抽样
            print("Stratified split not possible, using random split instead")
            train_idx, test_idx = train_test_split(
                range(len(dataset)),
                test_size=0.2,
                random_state=42
            )
        
        # 按索引获取训练/测试集
        train_ds = dataset[train_idx]
        test_ds = dataset[test_idx]
        
        # 打印数据集信息
        test_labels = np.array([data.y.item() for data in test_ds], dtype=np.int64)
        
        # 确保测试标签非负且在有效范围内
        if np.any(test_labels < 0) or np.any(test_labels >= num_classes):
            print(f"Warning: Test labels out of valid range [0, {num_classes-1}], converting to valid range")
            test_labels = np.clip(test_labels, 0, num_classes - 1)
        
        test_class_dist = np.bincount(test_labels, minlength=num_classes)
        print(f"Dataset size: {len(dataset)}, Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {num_classes}")
        print(f"Test set class distribution: {test_class_dist}")
        
        # 检查是否有足够的类别
        if len(np.unique(test_labels)) < 2:
            print("Warning: Only one class present in test set, metrics may be unreliable")

        # 遍历所有参数组合训练+测试
        for pool_name, pool in pooling_methods.items():
            for model_name, model_class in model_classes.items():
                for num_layers in num_layers_list:
                    for lr in lr_list:
                        for mode in modes:
                            try:
                                # 初始化模型
                                model = model_class(
                                    dataset.num_node_features, 
                                    num_classes, 
                                    pool, 
                                    num_layers
                                ).to(device)
                                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                                # 创建数据加载器
                                if mode == 'full':
                                    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                                else:
                                    train_loader = create_subgraph_loader(
                                        train_ds, 
                                        batch_size=16, 
                                        num_neighbors=[25, 10]
                                    )
                                    if train_loader is None:
                                        print(f"Skipping {model_name}-{pool_name}-Layers{num_layers}-LR{lr}-{mode}: No subgraphs")
                                        continue

                                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

                                # 训练模型
                                start_time = time.time()
                                for epoch in range(1, 11):  # 增加训练轮次
                                    loss = train(model, train_loader, optimizer, device, num_classes)
                                    if epoch % 10 == 0:
                                        print(f"[{dataset_name}-{model_name}-{pool_name}] Epoch {epoch}, Loss: {loss:.4f}")
                                training_time = time.time() - start_time

                                # 测试模型
                                acc, auc, ap = test(model, test_loader, device, num_classes, all_classes)
                                
                                # 输出结果
                                print(f'\nResult - Dataset: {dataset_name}, Model: {model_name}, Pooling: {pool_name}, '
                                      f'Layers: {num_layers}, LR: {lr}, Mode: {mode}\n'
                                      f'Accuracy: {acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, Time: {training_time:.2f}s\n')
                                
                            except Exception as e:
                                print(f"\nError with combination - Dataset: {dataset_name}, Model: {model_name}, "
                                      f"Pooling: {pool_name}, Layers: {num_layers}, LR: {lr}, Mode: {mode}")
                                print(f"Error message: {str(e)}\n")

if __name__ == "__main__":
    main()
    