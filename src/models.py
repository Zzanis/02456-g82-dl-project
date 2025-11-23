import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv


#-------------- Basic GCN ---------------
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x
    
#-------------- Advanced GCN with Regularization ---------------
class advanced_GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, dropout=0.3, use_layer_norm=False, use_residual=True):
        super(advanced_GCN, self).__init__()  # Fixed: was super(GCN, ...)
        
        # Graph convolution layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        
        # Projection layer for first residual connection (input dim != hidden dim)
        # This matches dimensions so we can add them: x (11 dims) + conv_out (64 dims) won't work
        # So we project x to 64 dims first
        self.input_proj = torch.nn.Linear(num_node_features, hidden_channels, bias=False)
        
        # Batch normalization (helps with training stability and speed)
        # Normalizes activations to have mean=0, std=1
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn5 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn6 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Optional: Layer normalization (alternative to batch norm)
        # More stable for small batches or varying graph sizes
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            from torch_geometric.nn import LayerNorm
            self.ln1 = LayerNorm(hidden_channels)
            self.ln2 = LayerNorm(hidden_channels)
            self.ln3 = LayerNorm(hidden_channels)
            self.ln4 = LayerNorm(hidden_channels)
            self.ln5 = LayerNorm(hidden_channels)
        
        # Dropout (randomly sets neurons to 0 during training)
        # Prevents overfitting by forcing network to learn redundant representations
        self.dropout = torch.nn.Dropout(dropout)
        
        # Whether to use residual connections
        self.use_residual = use_residual
        
        # Final prediction layer 
        self.linear = torch.nn.Linear(hidden_channels, 1)

        # Kaiming He initialization (for ReLU activations)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming He initialization for ReLU networks."""
        # Initialize linear layers with Kaiming He
        torch.nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)
        
        # GCNConv layers - initialize their internal linear layers
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            if hasattr(conv, 'lin'):
                torch.nn.init.kaiming_normal_(conv.lin.weight, mode='fan_in', nonlinearity='relu')
                if conv.lin.bias is not None:
                    torch.nn.init.zeros_(conv.lin.bias)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Store original input for first residual connection
        x_input = self.input_proj(x) if self.use_residual else None

        # Layer 1: Conv -> Norm -> Activation -> Dropout
        identity = x_input  # Save for residual
        x = self.conv1(x, edge_index)
        x = self.ln1(x, batch) if self.use_layer_norm else self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 2
        identity = x  # Save current x
        x = self.conv2(x, edge_index)
        x = self.ln2(x, batch) if self.use_layer_norm else self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 3
        identity = x
        x = self.conv3(x, edge_index)
        x = self.ln3(x, batch) if self.use_layer_norm else self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 4
        identity = x
        x = self.conv4(x, edge_index)
        x = self.ln4(x, batch) if self.use_layer_norm else self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 5
        identity = x
        x = self.conv5(x, edge_index)
        x = self.ln5(x, batch) if self.use_layer_norm else self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!


        # Readout: aggregate node features to graph-level
        x = global_mean_pool(x, batch)
        
        # Final prediction
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