import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv


#-------------- Basic GCN ---------------
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x

#-------------- Residual Block GCN ---------------
    
class GCNResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, p_drop: float = 0.0):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)
        self.bn = torch.nn.BatchNorm1d(out_ch)
        self.act = torch.nn.ReLU()
        self.has_proj = (in_ch != out_ch)
        self.proj = torch.nn.Linear(in_ch, out_ch, bias=False) if self.has_proj else None
        self.dropout = torch.nn.Dropout(p_drop) if p_drop and p_drop > 0.0 else None

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.has_proj:
            x = self.proj(x)
        return out + x  # Residual-Add
    
#-------------- GCN with Residual Blocks ---------------
class GCNResidual(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super().__init__()
        self.block1 = GCNResBlock(num_node_features, hidden_channels, p_drop=0.0)
        self.block2 = GCNResBlock(hidden_channels, hidden_channels, p_drop=0.0)
        self.block3 = GCNResBlock(hidden_channels, hidden_channels, p_drop=0.0)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x
    
#-------------- GCN with Residual Blocks and Dropout ---------------
    
class GCNResidualDropout(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, p_drop=0.1):
        super().__init__()
        self.block1 = GCNResBlock(num_node_features, hidden_channels, p_drop=p_drop)
        self.block2 = GCNResBlock(hidden_channels, hidden_channels, p_drop=p_drop)
        self.block3 = GCNResBlock(hidden_channels, hidden_channels, p_drop=p_drop)
        self.head_drop = torch.nn.Dropout(p_drop)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.head_drop(x)
        x = self.linear(x)
        return x
    
#-------------- GraphSAGE ---------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # SAGE Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)

        # SAGE Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop(x)

        # SAGE Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Pooling & Output
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x