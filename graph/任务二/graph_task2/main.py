# import os
# import time
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv
# from torch.nn import Sequential, Linear, ReLU
# from torch_geometric.datasets import Planetoid, Flickr
# from torch_geometric.transforms import Compose, NormalizeFeatures, RandomLinkSplit
# from torch_geometric.utils import negative_sampling, subgraph
# from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
# from torch.optim.lr_scheduler import StepLR
# from tqdm import tqdm
# import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
# from torch_geometric.loader import NeighborSampler


# # ---------------------- 1. 图神经网络模型定义 ----------------------
# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def encode(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

#     def decode(self, z, edge_label_index):
#         src = z[edge_label_index[0]]
#         dst = z[edge_label_index[1]]
#         return (src * dst).sum(dim=-1)

#     def forward(self, x, edge_index, edge_label_index):
#         z = self.encode(x, edge_index)
#         return self.decode(z, edge_label_index)


# class GAT(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=False)
#         self.conv2 = GATConv(hidden_channels, out_channels, heads=8, concat=False)

#     def encode(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

#     def decode(self, z, edge_label_index):
#         src = z[edge_label_index[0]]
#         dst = z[edge_label_index[1]]
#         return (src * dst).sum(dim=-1)

#     def forward(self, x, edge_index, edge_label_index):
#         z = self.encode(x, edge_index)
#         return self.decode(z, edge_label_index)


# class SAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, out_channels)

#     def encode(self, x, edge_index_or_adjs):
#         # 适配两种输入：全图edge_index / 采样后的adjs
#         if isinstance(edge_index_or_adjs, list):  # 邻居采样的adjs
#             # 假设adjs有两层，对应两层conv
#             adj1, adj2 = edge_index_or_adjs  # 注意：adjs是列表，长度为层数
#             x = F.relu(self.conv1(x, adj1.edge_index, size=adj1.size))
#             x = F.dropout(x, p=0.5, training=self.training)
#             x = self.conv2(x, adj2.edge_index, size=adj2.size)
#         else:  # 全图edge_index
#             x = F.relu(self.conv1(x, edge_index_or_adjs))
#             x = F.dropout(x, p=0.5, training=self.training)
#             x = self.conv2(x, edge_index_or_adjs)
#         return x

#     def decode(self, z, edge_label_index):
#         src = z[edge_label_index[0]]
#         dst = z[edge_label_index[1]]
#         return (src * dst).sum(dim=-1)

#     def forward(self, x, edge_index, edge_label_index):
#         z = self.encode(x, edge_index)
#         return self.decode(z, edge_label_index)


# class GIN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GINConv(Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)))
#         self.conv2 = GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels)))
#         self.dropout = nn.Dropout(0.5)

#     def encode(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)
#         return x

#     def decode(self, z, edge_label_index):
#         src = z[edge_label_index[0]]
#         dst = z[edge_label_index[1]]
#         return (src * dst).sum(dim=-1)

#     def forward(self, x, edge_index, edge_label_index):
#         z = self.encode(x, edge_index)
#         return self.decode(z, edge_label_index)


# # ---------------------- 2. 早停与工具函数 ----------------------
# class EarlyStopping:
#     def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint/'):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#         os.makedirs(path, exist_ok=True)

#     def __call__(self, val_loss, model, exp_name):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, exp_name)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             # if self.verbose:
#             #     print(f'EarlyStopping counter: {self.counter}/{self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, exp_name)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, exp_name):
#         # if self.verbose:
#         #     print(f'Val loss down ({self.val_loss_min:.6f}→{val_loss:.6f}), save model...')
#         torch.save(model.state_dict(), f'{self.path}/{exp_name}_best.pt')
#         self.val_loss_min = val_loss


# def setup_seed(seed=42):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


# def get_metrics(out, edge_label):
#     edge_label = edge_label.cpu().numpy()
#     out = out.cpu().numpy()
#     pred = (out > 0.5).astype(int)
#     auc = roc_auc_score(edge_label, out) if len(np.unique(edge_label)) > 1 else 0.0
#     f1 = f1_score(edge_label, pred) if len(np.unique(edge_label)) > 1 else 0.0
#     ap = average_precision_score(edge_label, out) if len(np.unique(edge_label)) > 1 else 0.0
#     return round(auc, 4), round(f1, 4), round(ap, 4)


# def sample_subgraph(data, sample_ratio=0.5):
#     """从全图中采样子图（按节点比例）"""
#     num_nodes = data.num_nodes
#     sample_size = int(num_nodes * sample_ratio)
#     sampled_nodes = torch.randperm(num_nodes)[:sample_size].to(data.x.device)

#     # 生成子图的边（只保留两端都在采样节点中的边）
#     edge_mask = torch.isin(data.edge_index[0], sampled_nodes) & torch.isin(data.edge_index[1], sampled_nodes)
#     sub_edge_index = data.edge_index[:, edge_mask]

#     # 调整节点索引（子图节点索引从0重新开始）
#     node_map = torch.full((num_nodes,), -1, device=data.x.device)
#     node_map[sampled_nodes] = torch.arange(sample_size, device=data.x.device)
#     sub_edge_index = node_map[sub_edge_index]

#     # 处理边标签索引
#     if hasattr(data, 'edge_label_index'):
#         label_mask = torch.isin(data.edge_label_index[0], sampled_nodes) & torch.isin(data.edge_label_index[1], sampled_nodes)
#         sub_edge_label_index = data.edge_label_index[:, label_mask]
#         sub_edge_label_index = node_map[sub_edge_label_index]
#         sub_edge_label = data.edge_label[label_mask]
#     else:
#         sub_edge_label_index = sub_edge_index
#         sub_edge_label = torch.ones(sub_edge_index.size(1), device=data.x.device)

#     return type(data)(
#         x=data.x[sampled_nodes],
#         edge_index=sub_edge_index,
#         edge_label_index=sub_edge_label_index,
#         edge_label=sub_edge_label,
#         num_nodes=sample_size
#     )


# # def load_dataset(dataset_name, root='./datasets/'):
# #     """加载数据集并分割为训练/验证/测试集"""
# #     os.makedirs(root, exist_ok=True)
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# #     if dataset_name in ['Cora', 'CiteSeer']:
# #         transform = Compose([
# #             NormalizeFeatures(),
# #             RandomLinkSplit(
# #                 num_val=0.1, num_test=0.1, is_undirected=True,
# #                 add_negative_train_samples=False
# #             )
# #         ])
# #         dataset = Planetoid(root=root, name=dataset_name, transform=transform)
# #         train_data, val_data, test_data = dataset[0]
# #         num_features = dataset.num_features
        
# #     elif dataset_name == 'Flickr':
# #         dataset = Flickr(root=root, transform=NormalizeFeatures())
# #         data = dataset[0]
# #         rls = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False)
# #         train_data, val_data, test_data = rls(data)
# #         num_features = dataset.num_features
        
# #     else:
# #         raise ValueError(f"未知数据集: {dataset_name} (支持 Cora/CiteSeer/Flickr)")
    
# #     return (
# #         train_data.to(device),
# #         val_data.to(device),
# #         test_data.to(device),
# #         num_features
# #     )

# # 替换原有的load_dataset函数为以下版本
# def load_dataset(dataset_name, root='./datasets/'):
#     """加载本地数据集并分割为训练/验证/测试集（适配旧版PyG，不使用download参数）"""
#     # 检查本地数据集根目录是否存在
#     if not os.path.exists(root):
#         raise FileNotFoundError(f"本地数据集根目录不存在：{root}\n请确保数据集文件夹结构正确")
    
#     # 检查当前数据集的raw文件夹是否存在
#     dataset_root = os.path.join(root, dataset_name)
#     raw_dir = os.path.join(dataset_root, 'raw')
#     if not os.path.exists(raw_dir):
#         raise FileNotFoundError(f"数据集 {dataset_name} 的raw文件夹不存在：{raw_dir}\n请放入对应的raw数据文件")

#     # 对于旧版PyG，通过提前创建processed文件夹避免触发下载
#     processed_dir = os.path.join(dataset_root, 'processed')
#     os.makedirs(processed_dir, exist_ok=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     if dataset_name in ['Cora', 'CiteSeer']:
#         # 旧版PyG的Planetoid不支持download参数，通过检查文件存在性确保不下载
#         transform = Compose([
#             NormalizeFeatures(),
#             RandomLinkSplit(
#                 num_val=0.1, num_test=0.1, is_undirected=True,
#                 add_negative_train_samples=False
#             )
#         ])
        
#         # 关键修改：移除download参数，通过提前准备文件避免下载
#         dataset = Planetoid(
#             root=root, 
#             name=dataset_name, 
#             transform=transform
#         )
#         train_data, val_data, test_data = dataset[0]
#         num_features = dataset.num_features
        
#     elif dataset_name == 'Flickr':
#         import pickle
#         import numpy as np
#         from torch_geometric.data import Data
#         data_dict = pickle.load(open(os.path.join(root, 'Flickr/flickr.pkl'), 'rb'))
#         edge_index = torch.tensor(np.array(data_dict['topo'].nonzero()), dtype=torch.long)
#         x = torch.tensor(data_dict['attr'].toarray(), dtype=torch.float) if 'attr' in data_dict else torch.eye(data_dict['topo'].shape[0])
#         y = torch.tensor(data_dict['label'], dtype=torch.long) if 'label' in data_dict else None
#         data = Data(x=x, edge_index=edge_index, y=y)
#         transform = Compose([
#             NormalizeFeatures(),
#             RandomLinkSplit(
#                 num_val=0.1, num_test=0.1, is_undirected=True,
#                 add_negative_train_samples=False
#             )
#         ])
#         train_data, val_data, test_data = transform(data)
#         num_features = data.num_features
        
#     else:
#         raise ValueError(f"未知数据集: {dataset_name} (支持 Cora/CiteSeer/Flickr)")
    
#     print(f"✅ 本地数据集 {dataset_name} 加载成功（节点数：{train_data.num_nodes}，特征维度：{num_features}）")
#     return (
#         train_data.to(device),
#         val_data.to(device),
#         test_data.to(device),
#         num_features
#     )
    

# # ---------------------- 3. 训练与测试函数 ----------------------
# def train_one_exp(model, train_data, val_data, test_data, lr, exp_name, epochs=10):
#     """单组实验的训练（全图或子图）"""
#     device = train_data.x.device
#     model = model.to(device)
#     torch.cuda.empty_cache()  # 清理 GPU 内存
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     criterion = nn.BCEWithLogitsLoss().to(device)
#     scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
#     early_stopping = EarlyStopping(patience=5, verbose=True)

#     start_time = time.time()
#     best_test_metrics = (0.0, 0.0, 0.0)

#     for epoch in range(epochs):  # 注意：移除了model.train()到循环外
#         torch.cuda.empty_cache()  # 每个epoch开始清空缓存

#         # 针对全图训练，初始化邻居采样器（子图训练可跳过）
#         if train_data.num_nodes > 10000 :  # 仅大型图启用（如Flickr）
#             sampler = NeighborSampler(
#                 train_data.edge_index,  # 全图边索引
#                 sizes=[10, 5],  # 每一层采样10、5个邻居（可调整）
#                 batch_size=512,  # 每批处理1024个节点（根据GPU内存调整）
#                 shuffle=True,

#             )
            
#             model.train()
#             total_loss = 0.0
#             # 遍历采样的节点批次
#             for batch_size, n_id, adjs in sampler:
#                 # adjs: 采样后的邻居结构（多层），需转为GPU格式
#                 adjs = [adj.to(device) for adj in adjs]
                
#                 optimizer.zero_grad()  # 每个批次优化
                
#                 # 1. 分批计算节点嵌入z（仅当前批次节点n_id的嵌入）
#                 z = model.encode(train_data.x[n_id], adjs)  # 需修改model.encode适配adjs
                
#                 # 生成当前批次的负样本和训练边（需调整为仅当前批次节点）
#                 # 注意：原负样本逻辑需适配批次，这里简化假设全图负样本，但实际需批次内生成
#                 neg_edge_index = negative_sampling(
#                     edge_index=adjs[-1].edge_index,  # 使用采样后的边
#                     num_nodes=batch_size,
#                     num_neg_samples=batch_size,
#                     method='sparse'
#                 ).to(device)
#                 train_edge_label_index = torch.cat([adjs[-1].edge_index, neg_edge_index], dim=-1)  # 简化，使用采样边作为正样本
#                 train_edge_label = torch.cat([torch.ones(adjs[-1].edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device)
                
#                 # 2. 从当前批次z中提取训练边的src/dst嵌入（调整索引到批次本地）
#                 local_node_map = {global_id.item(): local_idx for local_idx, global_id in enumerate(n_id)}
#                 src_idx = torch.tensor([local_node_map.get(idx.item(), -1) for idx in train_edge_label_index[0] if idx.item() in local_node_map], device=device)
#                 dst_idx = torch.tensor([local_node_map.get(idx.item(), -1) for idx in train_edge_label_index[1] if idx.item() in local_node_map], device=device)
#                 valid_mask = (src_idx >= 0) & (dst_idx >= 0)
#                 src = z[src_idx[valid_mask]]
#                 dst = z[dst_idx[valid_mask]]
#                 out = (src * dst).sum(dim=-1)  # 对应原decode逻辑
                
#                 # 3. 计算批次损失并反向传播
#                 batch_label = train_edge_label[valid_mask]
#                 if len(batch_label) > 0:
#                     loss = criterion(out, batch_label)
#                     loss.backward()
#                     optimizer.step()
#                     total_loss += loss.item() * len(batch_label)

#             # 平均损失（替代原单批次loss）
#             loss = total_loss / train_data.num_nodes if total_loss > 0 else 0.0
#         else:
#             # 原逻辑 for 小图
#             model.train()
#             # 生成负样本（正负样本1:1，但限制最大负样本数以防OOM）
#             num_neg_samples = min(train_data.edge_label_index.size(1), 10)  # 例如限制到10万
#             neg_edge_index = negative_sampling(
#                 edge_index=train_data.edge_index,
#                 num_nodes=train_data.num_nodes,
#                 num_neg_samples=num_neg_samples,
#                 method='sparse'
#             ).to(device)
#             train_edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
#             train_edge_label = torch.cat([train_data.edge_label, torch.zeros_like(neg_edge_index[0])], dim=0)

#             # 前向传播与优化
#             optimizer.zero_grad()
#             out = model(train_data.x, train_data.edge_index, train_edge_label_index).view(-1)
#             loss = criterion(out, train_edge_label)
#             loss.backward()
#             optimizer.step()

#         scheduler.step()

#         # 验证与测试（每5轮评估一次）
#         if (epoch + 1) % 5 == 0:
#             model.eval()
#             with torch.no_grad():
#                 # 验证集损失
#                 val_out = model(val_data.x, val_data.edge_index, val_data.edge_label_index).view(-1)
#                 val_loss = criterion(val_out, val_data.edge_label).item()

#                 # 测试集性能
#                 test_out = model(test_data.x, test_data.edge_index, test_data.edge_label_index).view(-1).sigmoid()
#                 test_auc, test_f1, test_ap = get_metrics(test_out, test_data.edge_label)

#             # 更新最优性能
#             if test_auc > best_test_metrics[0]:
#                 best_test_metrics = (test_auc, test_f1, test_ap)

#             # 早停检查
#             early_stopping(val_loss, model, exp_name)
#             if early_stopping.early_stop:
#                 print(f"早停于第 {epoch+1} 轮")
#                 break

#     # 计算训练时间
#     train_time = round(time.time() - start_time, 2)

#     # 加载最优模型并最终测试
#     model.load_state_dict(torch.load(f'checkpoint/{exp_name}_best.pt'))
#     model.eval()
#     with torch.no_grad():
#         test_out = model(test_data.x, test_data.edge_index, test_data.edge_label_index).view(-1).sigmoid()
#         final_auc, final_f1, final_ap = get_metrics(test_out, test_data.edge_label)

#     return final_auc, final_f1, final_ap, train_time


# # ---------------------- 4. 主实验流程 ----------------------
# def run_all_experiments():
#     # 实验配置
#     setup_seed(42)
#     # datasets = ['Cora', 'CiteSeer', 'Flickr']
#     datasets = ['Flickr']
#     models = {
#         'GCN': GCN,
#         'GAT': GAT,
#         'SAGE': SAGE,
#         'GIN': GIN
#     }
#     lrs = [0.01, 0.03, 0.06]
#     subgraph_ratio = 0.5  # 子图采样比例
#     hidden_channels = 16
#     out_channels = 16
#     results = []  # 存储所有实验结果

#     # 创建结果保存目录
#     os.makedirs('results', exist_ok=True)

#     # 遍历所有实验组合
#     for dataset_name in datasets:
#         print(f"\n{'='*50}")
#         print(f"开始数据集 {dataset_name} 的实验")
#         print(f"{'='*50}")
        
#         # 加载数据集
#         train_full, val_data, test_data, num_features = load_dataset(dataset_name)
#         # 采样子图训练数据
#         train_sub = sample_subgraph(train_full, sample_ratio=subgraph_ratio)
#         print(f"全图训练数据 - 节点数: {train_full.num_nodes}, 边数: {train_full.edge_index.size(1)}")
#         print(f"子图训练数据 - 节点数: {train_sub.num_nodes}, 边数: {train_sub.edge_index.size(1)}")

#         for model_name, ModelClass in models.items():
#             for lr in lrs:
#                 # 1. 全图训练实验
#                 exp_name_full = f"{dataset_name}_{model_name}_lr{lr}_full"
#                 print(f"\n--- 开始实验: {exp_name_full} ---")
#                 model_full = ModelClass(num_features, hidden_channels, out_channels)
#                 auc_full, f1_full, ap_full, time_full = train_one_exp(
#                     model=model_full,
#                     train_data=train_full,
#                     val_data=val_data,
#                     test_data=test_data,
#                     lr=lr,
#                     exp_name=exp_name_full
#                 )

#                 # 2. 子图训练实验
#                 exp_name_sub = f"{dataset_name}_{model_name}_lr{lr}_sub"
#                 print(f"\n--- 开始实验: {exp_name_sub} ---")
#                 model_sub = ModelClass(num_features, hidden_channels, out_channels)
#                 auc_sub, f1_sub, ap_sub, time_sub = train_one_exp(
#                     model=model_sub,
#                     train_data=train_sub,
#                     val_data=val_data,
#                     test_data=test_data,
#                     lr=lr,
#                     exp_name=exp_name_sub
#                 )

#                 # 保存结果
#                 results.append({
#                     '数据集': dataset_name,
#                     '模型': model_name,
#                     '学习率': lr,
#                     '训练模式': '全图',
#                     'AUC': auc_full,
#                     'F1': f1_full,
#                     'AP': ap_full,
#                     '训练时间(秒)': time_full
#                 })
#                 results.append({
#                     '数据集': dataset_name,
#                     '模型': model_name,
#                     '学习率': lr,
#                     '训练模式': '子图',
#                     'AUC': auc_sub,
#                     'F1': f1_sub,
#                     'AP': ap_sub,
#                     '训练时间(秒)': time_sub
#                 })

#                 # 打印当前实验对比
#                 print(f"\n{exp_name_full} vs {exp_name_sub} 对比:")
#                 print(f"全图 - AUC: {auc_full}, F1: {f1_full}, AP: {ap_full}, 时间: {time_full}s")
#                 print(f"子图 - AUC: {auc_sub}, F1: {f1_sub}, AP: {ap_sub}, 时间: {time_sub}s")
#                 print(f"性能变化 - AUC: {round(auc_sub-auc_full, 4)}, 时间变化: {round(time_sub-time_full, 2)}s")

#     # 保存所有结果到CSV
#     results_df = pd.DataFrame(results)
#     results_df.to_csv('results/experiment_results.csv', index=False, encoding='utf-8-sig')
#     print("\n所有实验完成！结果已保存到 results/experiment_results.csv")

#     # 打印汇总表格（按数据集和模型分组的最佳结果）
#     print("\n" + "="*80)
#     print("实验结果汇总（最佳AUC）")
#     print("="*80)
#     for dataset in datasets:
#         for model in models.keys():
#             subset = results_df[(results_df['数据集'] == dataset) & (results_df['模型'] == model)]
#             best_full = subset[subset['训练模式'] == '全图'].sort_values('AUC', ascending=False).iloc[0]
#             best_sub = subset[subset['训练模式'] == '子图'].sort_values('AUC', ascending=False).iloc[0]
            
#             print(f"{dataset} - {model}:")
#             print(f"  全图最佳 (lr={best_full['学习率']}): AUC={best_full['AUC']}, 时间={best_full['训练时间(秒)']}s")
#             print(f"  子图最佳 (lr={best_sub['学习率']}): AUC={best_sub['AUC']}, 时间={best_sub['训练时间(秒)']}s")
#             print(f"  差异: AUC变化{round(best_sub['AUC']-best_full['AUC'], 4)}, 时间变化{round(best_sub['训练时间(秒)']-best_full['训练时间(秒)'], 2)}s")
#             print("-"*60)


# if __name__ == "__main__":
#     run_all_experiments()

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomLinkSplit
from torch_geometric.utils import negative_sampling, subgraph, remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.loader import NeighborSampler


# ---------------------- 1. 图神经网络模型定义 ----------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, out_channels, heads=4, concat=False)
        self.dropout = nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def encode(self, x, edge_index_or_adjs):
        # 适配两种输入：全图edge_index / 采样后的adjs
        if isinstance(edge_index_or_adjs, list):  # 邻居采样的adjs
            # 处理邻居采样的情况
            x_target = x[:edge_index_or_adjs[0].size[1]]  # 只保留目标节点
            
            # 第一层卷积
            x = F.relu(self.conv1((x, x_target), edge_index_or_adjs[0].edge_index))
            x = self.dropout(x)
            
            # 更新目标节点用于第二层
            x_target = x[:edge_index_or_adjs[1].size[1]]
            
            # 第二层卷积
            x = F.relu(self.conv2((x, x_target), edge_index_or_adjs[1].edge_index))
            x = self.dropout(x)
            
            # 更新目标节点用于第三层
            x_target = x[:edge_index_or_adjs[2].size[1]] if len(edge_index_or_adjs) > 2 else x
            
            # 第三层卷积
            x = self.conv3((x, x_target), edge_index_or_adjs[-1].edge_index)
            return x
        else:  # 全图edge_index
            x = F.relu(self.conv1(x, edge_index_or_adjs))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index_or_adjs))
            x = self.dropout(x)
            x = self.conv3(x, edge_index_or_adjs)
            return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)))
        self.conv2 = GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)))
        self.conv3 = GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels)))
        self.dropout = nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


# ---------------------- 2. 早停与工具函数 ----------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint/'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        os.makedirs(path, exist_ok=True)

    def __call__(self, val_loss, model, exp_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, exp_name)
         
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, exp_name)
       
            self.counter = 0

    def save_checkpoint(self, val_loss, model, exp_name):
        torch.save(model.state_dict(), f'{self.path}/{exp_name}_best.pt')
        self.val_loss_min = val_loss


def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_metrics(out, edge_label):
    edge_label = edge_label.cpu().numpy()
    out = out.cpu().numpy()
    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out) if len(np.unique(edge_label)) > 1 else 0.0
    f1 = f1_score(edge_label, pred) if len(np.unique(edge_label)) > 1 else 0.0
    ap = average_precision_score(edge_label, out) if len(np.unique(edge_label)) > 1 else 0.0
    return round(auc, 4), round(f1, 4), round(ap, 4)


def sample_subgraph(data, sample_ratio=0.5):
    """从全图中采样子图（按节点比例）"""
    num_nodes = data.num_nodes
    sample_size = int(num_nodes * sample_ratio)
    sampled_nodes = torch.randperm(num_nodes)[:sample_size].to(data.x.device)

    # 生成子图的边（只保留两端都在采样节点中的边）
    edge_mask = torch.isin(data.edge_index[0], sampled_nodes) & torch.isin(data.edge_index[1], sampled_nodes)
    sub_edge_index = data.edge_index[:, edge_mask]

    # 调整节点索引（子图节点索引从0重新开始）
    node_map = torch.full((num_nodes,), -1, device=data.x.device)
    node_map[sampled_nodes] = torch.arange(sample_size, device=data.x.device)
    sub_edge_index = node_map[sub_edge_index]

    # 处理边标签索引
    if hasattr(data, 'edge_label_index'):
        label_mask = torch.isin(data.edge_label_index[0], sampled_nodes) & torch.isin(data.edge_label_index[1], sampled_nodes)
        sub_edge_label_index = data.edge_label_index[:, label_mask]
        sub_edge_label_index = node_map[sub_edge_label_index]
        sub_edge_label = data.edge_label[label_mask]
    else:
        sub_edge_label_index = sub_edge_index
        sub_edge_label = torch.ones(sub_edge_index.size(1), device=data.x.device)

    return type(data)(
        x=data.x[sampled_nodes],
        edge_index=sub_edge_index,
        edge_label_index=sub_edge_label_index,
        edge_label=sub_edge_label,
        num_nodes=sample_size
    )


def load_dataset(dataset_name, root='./datasets/'):
    """加载本地数据集并分割为训练/验证/测试集（适配旧版PyG，不使用download参数）"""
    if not os.path.exists(root):
        raise FileNotFoundError(f"本地数据集根目录不存在：{root}\n请确保数据集文件夹结构正确")
    
    dataset_root = os.path.join(root, dataset_name)
    raw_dir = os.path.join(dataset_root, 'raw')
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"数据集 {dataset_name} 的raw文件夹不存在：{raw_dir}\n请放入对应的raw数据文件")

    processed_dir = os.path.join(dataset_root, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset_name in ['Cora', 'CiteSeer']:
        transform = Compose([
            NormalizeFeatures(),
            RandomLinkSplit(
                num_val=0.1, num_test=0.1, is_undirected=True,
                add_negative_train_samples=False
            )
        ])
        
        dataset = Planetoid(
            root=root, 
            name=dataset_name, 
            transform=transform
        )
        train_data, val_data, test_data = dataset[0]
        num_features = dataset.num_features
        
    elif dataset_name == 'Flickr':
        import pickle
        import numpy as np
        from torch_geometric.data import Data
        data_dict = pickle.load(open(os.path.join(root, 'Flickr/flickr.pkl'), 'rb'))
        edge_index = torch.tensor(np.array(data_dict['topo'].nonzero()), dtype=torch.long)
        x = torch.tensor(data_dict['attr'].toarray(), dtype=torch.float) if 'attr' in data_dict else torch.eye(data_dict['topo'].shape[0])
        y = torch.tensor(data_dict['label'], dtype=torch.long) if 'label' in data_dict else None
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # 对Flickr数据集进行整体采样，只使用1/10的数据
        sample_ratio = 0.1
        num_nodes = data.num_nodes
        sample_size = int(num_nodes * sample_ratio)
        sampled_nodes = torch.randperm(num_nodes)[:sample_size]
        
        # 筛选采样节点相关的边
        edge_mask = torch.isin(data.edge_index[0], sampled_nodes) & torch.isin(data.edge_index[1], sampled_nodes)
        sampled_edge_index = data.edge_index[:, edge_mask]
        
        # 调整节点索引
        node_map = torch.full((num_nodes,), -1)
        node_map[sampled_nodes] = torch.arange(sample_size)
        sampled_edge_index = node_map[sampled_edge_index]
        
        # 创建采样后的数据集
        sampled_data = Data(
            x=data.x[sampled_nodes],
            edge_index=sampled_edge_index,
            y=data.y[sampled_nodes] if data.y is not None else None,
            num_nodes=sample_size
        )
        
        # 分割数据集
        transform = Compose([
            NormalizeFeatures(),
            RandomLinkSplit(
                num_val=0.1, num_test=0.1, is_undirected=True,
                add_negative_train_samples=False
            )
        ])
        train_data, val_data, test_data = transform(sampled_data)
        num_features = sampled_data.num_features
        
        print(f"已对Flickr数据集进行采样，保留 {sample_ratio*100}% 的数据")
        print(f"采样后节点数: {sample_size}, 边数: {sampled_edge_index.size(1)}")
        
    else:
        raise ValueError(f"未知数据集: {dataset_name} (支持 Cora/CiteSeer/Flickr)")
    
    # 添加自环以提高模型稳定性
    train_data.edge_index, _ = add_self_loops(train_data.edge_index)
    val_data.edge_index, _ = add_self_loops(val_data.edge_index)
    test_data.edge_index, _ = add_self_loops(test_data.edge_index)
    
    print(f"✅ 本地数据集 {dataset_name} 加载成功（节点数：{train_data.num_nodes}，特征维度：{num_features}）")
    return (
        train_data.to(device),
        val_data.to(device),
        test_data.to(device),
        num_features
    )
    

# ---------------------- 3. 训练与测试函数 ----------------------
def train_one_exp(model, train_data, val_data, test_data, lr, exp_name, epochs=100):
    """单组实验的训练（全图或子图）"""
    device = train_data.x.device
    model = model.to(device)
    torch.cuda.empty_cache()  # 清理 GPU 内存
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss().to(device)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    start_time = time.time()
    best_test_metrics = (0.0, 0.0, 0.0)

    # 检查是否是Flickr数据集的SAGE模型全图训练
    is_flickr_sage_full = "Flickr" in exp_name and "SAGE" in exp_name and "full" in exp_name
    
    # 对于Flickr SAGE全图训练，强制使用更严格的内存优化策略
    if is_flickr_sage_full:
        print("启用Flickr SAGE全图训练的内存优化策略")
        # 强制使用邻居采样，即使节点数判断不触发
        use_sampler = True
        # 减小批次大小以减少内存占用
        batch_size = 256  # 从512减小到256
        # 调整采样邻居数量
        sample_sizes = [5, 3, 2]  # 增加一层采样以匹配三层模型
    else:
        # 原始逻辑
        use_sampler = train_data.num_nodes > 10000
        batch_size = 512
        sample_sizes = [10, 5, 3]  # 增加一层采样以匹配三层模型

    for epoch in range(epochs):
        torch.cuda.empty_cache()  # 每个epoch开始清空缓存

        # 邻居采样训练（针对大型图或强制启用的情况）
        if use_sampler:
            sampler = NeighborSampler(
                train_data.edge_index,
                sizes=sample_sizes,
                batch_size=batch_size,
                shuffle=True,
            )
            
            model.train()
            total_loss = 0.0
            batch_count = 0
            
            # 遍历采样的节点批次
            for batch_size, n_id, adjs in sampler:
                adjs = [adj.to(device) for adj in adjs]
                
                optimizer.zero_grad()
                
                # 计算节点嵌入z
                z = model.encode(train_data.x[n_id], adjs)
                
                # 生成当前批次的负样本和训练边
                # 确保负样本不是真实存在的边
                neg_edge_index = negative_sampling(
                    edge_index=adjs[-1].edge_index,
                    num_nodes=batch_size,
                    num_neg_samples=min(adjs[-1].edge_index.size(1), batch_size),  # 负样本数量与正样本相当
                    method='sparse'
                ).to(device)
                
                # 组合正负样本
                train_edge_label_index = torch.cat([adjs[-1].edge_index, neg_edge_index], dim=-1)
                train_edge_label = torch.cat([
                    torch.ones(adjs[-1].edge_index.size(1)), 
                    torch.zeros(neg_edge_index.size(1))
                ], dim=0).to(device)
                
                # 提取训练边的src/dst嵌入（调整索引到批次本地）
                local_node_map = {global_id.item(): local_idx for local_idx, global_id in enumerate(n_id)}
                
                # 收集有效边的索引和对应的标签索引
                valid_indices = []
                src_list = []
                dst_list = []
                
                for i, (src_global, dst_global) in enumerate(zip(train_edge_label_index[0], train_edge_label_index[1])):
                    src_local = local_node_map.get(src_global.item(), -1)
                    dst_local = local_node_map.get(dst_global.item(), -1)
                    # 只保留两个索引都有效的边
                    if src_local != -1 and dst_local != -1 and src_local < len(z) and dst_local < len(z):
                        valid_indices.append(i)
                        src_list.append(src_local)
                        dst_list.append(dst_local)
                
                # 转换为张量
                src_idx = torch.tensor(src_list, device=device)
                dst_idx = torch.tensor(dst_list, device=device)
                
                # 获取对应的标签
                batch_label = train_edge_label[valid_indices].to(device)
                
                # 计算输出
                if len(src_idx) > 0 and len(dst_idx) > 0 and len(src_idx) == len(dst_idx):
                    src = z[src_idx]
                    dst = z[dst_idx]
                    out = (src * dst).sum(dim=-1)
                    
                    # 计算批次损失并反向传播
                    loss = criterion(out, batch_label)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_label)
                    batch_count += 1
                else:
                    # 如果没有有效边，跳过当前批次
                    continue

            # 计算平均损失
            loss = total_loss / batch_count if batch_count > 0 else 0.0
        else:
            # 小图训练逻辑
            model.train()
            # 生成负样本（数量与正样本相当）
            num_neg_samples = train_data.edge_label_index.size(1)
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index,
                num_nodes=train_data.num_nodes,
                num_neg_samples=num_neg_samples,
                method='sparse'
            ).to(device)
            train_edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
            train_edge_label = torch.cat([train_data.edge_label, torch.zeros_like(neg_edge_index[0])], dim=0)

            # 前向传播与优化
            optimizer.zero_grad()
            out = model(train_data.x, train_data.edge_index, train_edge_label_index).view(-1)
            loss = criterion(out, train_edge_label)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 每轮都进行验证，而不仅仅是每5轮
        model.eval()
        with torch.no_grad():
            # 验证集损失
            val_out = model(val_data.x, val_data.edge_index, val_data.edge_label_index).view(-1)
            val_loss = criterion(val_out, val_data.edge_label).item()

            # 测试集性能
            test_out = model(test_data.x, test_data.edge_index, test_data.edge_label_index).view(-1).sigmoid()
            test_auc, test_f1, test_ap = get_metrics(test_out, test_data.edge_label)

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test AUC: {test_auc:.4f}')

        # 更新最优性能
        if test_auc > best_test_metrics[0]:
            best_test_metrics = (test_auc, test_f1, test_ap)

        # 早停检查
        early_stopping(val_loss, model, exp_name)
        if early_stopping.early_stop:
            print(f"早停于第 {epoch+1} 轮")
            break

    # 计算训练时间
    train_time = round(time.time() - start_time, 2)

    # 加载最优模型并最终测试
    model.load_state_dict(torch.load(f'checkpoint/{exp_name}_best.pt'))
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x, test_data.edge_index, test_data.edge_label_index).view(-1).sigmoid()
        final_auc, final_f1, final_ap = get_metrics(test_out, test_data.edge_label)

    return final_auc, final_f1, final_ap, train_time


# ---------------------- 4. 主实验流程 ----------------------
def run_all_experiments():
    # 实验配置
    setup_seed(42)
    datasets = ['Flickr']
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'SAGE': SAGE,
        'GIN': GIN
    }
    lrs = [0.001, 0.005, 0.01]  # 调整学习率范围
    subgraph_ratio = 0.5  # 子图采样比例
    hidden_channels = 64  # 增加隐藏层维度
    out_channels = 32
    results = []  # 存储所有实验结果

    # 创建结果保存目录
    os.makedirs('results', exist_ok=True)

    # 遍历所有实验组合
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"开始数据集 {dataset_name} 的实验")
        print(f"{'='*50}")
        
        # 加载数据集
        train_full, val_data, test_data, num_features = load_dataset(dataset_name)
        # 采样子图训练数据
        train_sub = sample_subgraph(train_full, sample_ratio=subgraph_ratio)
        print(f"全图训练数据 - 节点数: {train_full.num_nodes}, 边数: {train_full.edge_index.size(1)}")
        print(f"子图训练数据 - 节点数: {train_sub.num_nodes}, 边数: {train_sub.edge_index.size(1)}")

        for model_name, ModelClass in models.items():
            for lr in lrs:
                # 1. 全图训练实验
                exp_name_full = f"{dataset_name}_{model_name}_lr{lr}_full"
                print(f"\n--- 开始实验: {exp_name_full} ---")
                model_full = ModelClass(num_features, hidden_channels, out_channels)
                auc_full, f1_full, ap_full, time_full = train_one_exp(
                    model=model_full,
                    train_data=train_full,
                    val_data=val_data,
                    test_data=test_data,
                    lr=lr,
                    exp_name=exp_name_full,
                    epochs=10  # 增加训练轮次
                )

                # 2. 子图训练实验
                exp_name_sub = f"{dataset_name}_{model_name}_lr{lr}_sub"
                print(f"\n--- 开始实验: {exp_name_sub} ---")
                model_sub = ModelClass(num_features, hidden_channels, out_channels)
                auc_sub, f1_sub, ap_sub, time_sub = train_one_exp(
                    model=model_sub,
                    train_data=train_sub,
                    val_data=val_data,
                    test_data=test_data,
                    lr=lr,
                    exp_name=exp_name_sub,
                    epochs=10  # 增加训练轮次
                )

                # 保存结果
                results.append({
                    '数据集': dataset_name,
                    '模型': model_name,
                    '学习率': lr,
                    '训练模式': '全图',
                    'AUC': auc_full,
                    'F1': f1_full,
                    'AP': ap_full,
                    '训练时间(秒)': time_full
                })
                results.append({
                    '数据集': dataset_name,
                    '模型': model_name,
                    '学习率': lr,
                    '训练模式': '子图',
                    'AUC': auc_sub,
                    'F1': f1_sub,
                    'AP': ap_sub,
                    '训练时间(秒)': time_sub
                })

                # 打印当前实验对比
                print(f"\n{exp_name_full} vs {exp_name_sub} 对比:")
                print(f"全图 - AUC: {auc_full}, F1: {f1_full}, AP: {ap_full}, 时间: {time_full}s")
                print(f"子图 - AUC: {auc_sub}, F1: {f1_sub}, AP: {ap_sub}, 时间: {time_sub}s")
                print(f"性能变化 - AUC: {round(auc_sub-auc_full, 4)}, 时间变化: {round(time_sub-time_full, 2)}s")

    # 保存所有结果到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/experiment_results.csv', index=False, encoding='utf-8-sig')
    print("\n所有实验完成！结果已保存到 results/experiment_results.csv")

    # 打印汇总表格
    print("\n" + "="*80)
    print("实验结果汇总（最佳AUC）")
    print("="*80)
    for dataset in datasets:
        for model in models.keys():
            subset = results_df[(results_df['数据集'] == dataset) & (results_df['模型'] == model)]
            best_full = subset[subset['训练模式'] == '全图'].sort_values('AUC', ascending=False).iloc[0]
            best_sub = subset[subset['训练模式'] == '子图'].sort_values('AUC', ascending=False).iloc[0]
            
            print(f"{dataset} - {model}:")
            print(f"  全图最佳 (lr={best_full['学习率']}): AUC={best_full['AUC']}, 时间={best_full['训练时间(秒)']}s")
            print(f"  子图最佳 (lr={best_sub['学习率']}): AUC={best_sub['AUC']}, 时间={best_sub['训练时间(秒)']}s")
            print(f"  差异: AUC变化{round(best_sub['AUC']-best_full['AUC'], 4)}, 时间变化{round(best_sub['训练时间(秒)']-best_full['训练时间(秒)'], 2)}s")
            print("-"*60)


if __name__ == "__main__":
    run_all_experiments()
    

