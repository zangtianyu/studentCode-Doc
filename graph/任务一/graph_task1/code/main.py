from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import NormalizeFeatures
from models import GCN, GAT, GraphSAGE, GIN
import torch.nn.functional as F
import torch
import train
from train import train 
import test
from test import test
import time 
from torch_geometric.loader import NeighborSampler


# 定义数据集、模型和学习率列表
datasets = ['Cora', 'CiteSeer', 'Flickr']
models_list = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
learning_rates = [0.01, 0.03, 0.06]
epochs = 11
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义train_with_sampling函数
def train_with_sampling(model, data, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs[0].edge_index)
        loss = F.nll_loss(out[:batch_size], data.y[n_id[:batch_size]].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
    return total_loss / len(train_loader.dataset)


for ds_name in datasets:
    # 加载数据集
    if ds_name in ['Cora', 'CiteSeer']:
        dataset = Planetoid(root=f'./dataset/{ds_name}', name=ds_name, transform=NormalizeFeatures())
    elif ds_name == 'Flickr':
        dataset = Flickr(root='./dataset/Flickr')
    data = dataset[0]

    for model_name in models_list:
        for lr in learning_rates:
            # 实例化模型
            if model_name == 'GCN':
                model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
            elif model_name == 'GAT':
                model = GAT(dataset.num_node_features, dataset.num_classes).to(device)
            elif model_name == 'GraphSAGE':
                model = GraphSAGE(dataset.num_node_features, dataset.num_classes).to(device)
            elif model_name == 'GIN':
                model = GIN(dataset.num_node_features, dataset.num_classes).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            print(f'\nDataset: {ds_name}, Model: {model_name}, LR: {lr}')

            # 全图训练
            start_time = time.time()
            for epoch in range(1, epochs):
                loss = train(model, data, optimizer, device)
                if epoch % 10 == 0:
                    accuracy = test(model, data, device)
                    print(f'Full Graph - Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
            full_graph_time = time.time() - start_time
            print(f'Full graph training time: {full_graph_time:.4f} seconds')

            # 重置模型和优化器以进行子图采样训练
            if model_name == 'GCN':
                model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
            elif model_name == 'GAT':
                model = GAT(dataset.num_node_features, dataset.num_classes).to(device)
            elif model_name == 'GraphSAGE':
                model = GraphSAGE(dataset.num_node_features, dataset.num_classes).to(device)
            elif model_name == 'GIN':
                model = GIN(dataset.num_node_features, dataset.num_classes).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            # 子图采样
            train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[15, 10], batch_size=128, shuffle=True, num_nodes=data.num_nodes)

            start_time = time.time()
            for epoch in range(1, epochs):
                loss = train_with_sampling(model, data, train_loader, optimizer, device)
                if epoch % 10 == 0:
                    accuracy = test(model, data, device)
                    print(f'Subgraph Sampling - Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
            subgraph_time = time.time() - start_time
            print(f'Subgraph sampling training time: {subgraph_time:.4f} seconds')

            # 比较
            print(f'Comparison: Full Graph Time: {full_graph_time:.4f}s, Subgraph Time: {subgraph_time:.4f}s')
            print(f'Time Difference: {full_graph_time - subgraph_time:.4f}s')






# 使用NeighborSampler进行子图采样
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[15, 10], batch_size=128, shuffle=True, num_nodes=data.num_nodes)

# 子图采样训练
start_time = time.time()
for epoch in range(1, epochs):
    loss = train_with_sampling(model, data, train_loader, optimizer, device)
    if epoch % 10 == 0:
        accuracy = test(model, data, device)
        print(f'(Subgraph Sampling) Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
subgraph_sampling_training_time = time.time() - start_time

print(f'Subgraph sampling training time: {subgraph_sampling_training_time:.4f} seconds')


