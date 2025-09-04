

from torch_geometric.nn import GCNConv,GATConv, SAGEConv, GINConv
import torch.nn.functional as F
import torch
from torch.nn import Sequential, Linear, ReLU


# GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

# GAT模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, concat=True)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# GraphSAGE模型
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    


# GIN模型
class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GIN, self).__init__()
        nn1 = Sequential(Linear(in_channels, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2)
        self.fc = Linear(64, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)