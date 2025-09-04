import json

# 读取原notebook
with open('graph-task2.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 新的数据加载代码
new_code = '''import pickle
import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 直接加载本地数据集
data_dict = pickle.load(open('dataset/CiteSeer/citeseer.pkl', 'rb'))
print(f'数据集名称: {data_dict["name"]}')
print(f'节点数: {data_dict["topo"].shape[0]}')
print(f'特征维度: {data_dict["attr"].shape[1]}')

# 转换为torch tensors
edge_index = torch.tensor(np.array(data_dict['topo'].nonzero()), dtype=torch.long)
x = torch.tensor(data_dict['attr'].toarray(), dtype=torch.float)
y = torch.tensor(data_dict['label'], dtype=torch.long)

# 创建PyTorch Geometric Data对象
data = Data(x=x, edge_index=edge_index, y=y)

# 应用变换
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])

# 应用变换
dataset = [data]  # 包装成数据集格式
train_data, val_data, test_data = transform(dataset)

print(f'转换完成! 节点数: {data.num_nodes}, 边数: {data.num_edges}, 特征维度: {data.num_features}')

# 确保数据在正确设备上
train_data = train_data.to(device)
val_data = val_data.to(device)  
test_data = test_data.to(device)

# 创建模型
model = GCN(data.num_features, 64, 128).to(device)
# model = GAT(data.num_features, 64, 128).to(device)
# model = SAGE(data.num_features, 64, 128).to(device)
# model = GIN(data.num_features, 64, 128).to(device)

# 训练模型
final_auc, final_ap = train(model, train_data, val_data, test_data, device)
print(f'最终测试结果 - AUC: {final_auc:.4f}, AP: {final_ap:.4f}')

# 测试模型
val_loss, test_auc, test_ap = test(model, val_data, test_data)
print(f'验证损失: {val_loss:.4f}, 测试AUC: {test_auc:.4f}, 测试AP: {test_ap:.4f}')'''

# 替换最后一个cell的source（应该是第6个cell，索引5）
notebook['cells'][5]['source'] = [new_code]

# 保存修改后的notebook
with open('graph-task2_local.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("成功修改graph_task2 notebook!")
