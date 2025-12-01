import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, DenseSAGEConv, radius_graph, MessagePassing, GraphNorm
from torch_geometric.utils import add_self_loops



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


class GraphSAGE(nn.Module):
    def __init__(
        self, 
        num_node_features, 
        hidden_channels=256,  # increased width
        num_layers=5,          # increased depth
        aggr='max', 
        dropout=0.2            # slightly higher dropout for stability
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Dynamically create layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(num_node_features, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final regression layer
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # first layer: project to hidden_channels
        x = F.relu(self.bns[0](self.convs[0](x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # subsequent layers with residual
        for conv, bn in zip(self.convs[1:], self.bns[1:]):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual in hidden space

        # Global pooling
        x = global_mean_pool(x, batch)

        # Regression output
        x = self.lin(x)
        return x


class SchNetLikeConv(MessagePassing):
    """
    Simplified SchNet-style continuous filter convolution
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__(aggr='add')
        self.lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.lin(x_j)


class SchNetLikeModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.1):
        super().__init__()
        self.dropout = dropout

         # Message-passing layers
        self.conv1 = SchNetLikeConv(num_node_features, hidden_channels)
        self.conv2 = SchNetLikeConv(hidden_channels, hidden_channels)
        self.conv3 = SchNetLikeConv(hidden_channels, hidden_channels)
        self.conv4 = SchNetLikeConv(hidden_channels, hidden_channels)  
        self.conv5 = SchNetLikeConv(hidden_channels, hidden_channels) 

        # # BatchNorm layers
        # self.bn1 = nn.BatchNorm1d(hidden_channels)
        # self.bn2 = nn.BatchNorm1d(hidden_channels)
        # self.bn3 = nn.BatchNorm1d(hidden_channels)
        # self.bn4 = nn.BatchNorm1d(hidden_channels)
        # self.bn5 = nn.BatchNorm1d(hidden_channels)

        # GraphNorm layers
        self.gn1 = GraphNorm(hidden_channels)
        self.gn2 = GraphNorm(hidden_channels)
        self.gn3 = GraphNorm(hidden_channels)
        self.gn4 = GraphNorm(hidden_channels)
        self.gn5 = GraphNorm(hidden_channels)

        # Final linear layer for regression
        self.linear = nn.Linear(hidden_channels, 1)

        # Final linear layer for regression
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

         # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = self.gn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2 with residual
        h2 = self.conv2(h1, edge_index)
        h2 = self.gn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + h1  # residual connection

        # Layer 3 with residual
        h3 = self.conv3(h2, edge_index)
        h3 = self.gn3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        h3 = h3 + h2  # residual connection

        # Layer 4 with residual
        h4 = self.conv4(h3, edge_index)
        h4 = self.gn4(h4)
        h4 = F.relu(h4)
        h4 = F.dropout(h4, p=self.dropout, training=self.training)
        h4 = h4 + h3  # residual connection

        # Layer 5 with residual
        h5 = self.conv5(h4, edge_index)
        h5 = self.gn5(h5)
        h5 = F.relu(h5)
        h5 = F.dropout(h5, p=self.dropout, training=self.training)
        h5 = h5 + h4  # residual connection

        # Global mean pooling
        x = global_mean_pool(h5, batch)

        # Final regression output
        x = self.linear(x)
        return x

#-------------- Advanced GCN with Regularization ---------------
class advanced_GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.05, use_layer_norm=False, use_graph_norm=True, use_residual=True, use_kaiming_init=False):
        super(advanced_GCN, self).__init__()  # Fixed: was super(GCN, ...)
        
        # Graph convolution layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        
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

        self.use_graph_norm = use_graph_norm
        if use_graph_norm:
            # from torch_geometric.nn import LayerNorm
            self.ln1 = GraphNorm(hidden_channels)
            self.ln2 = GraphNorm(hidden_channels)
            self.ln3 = GraphNorm(hidden_channels)
            self.ln4 = GraphNorm(hidden_channels)
            self.ln5 = GraphNorm(hidden_channels)
        
        # Dropout (randomly sets neurons to 0 during training)
        # Prevents overfitting by forcing network to learn redundant representations
        self.dropout = torch.nn.Dropout(dropout)
        
        # Whether to use residual connections
        self.use_residual = use_residual
        
        # Final prediction layer 
        self.linear = torch.nn.Linear(hidden_channels, 1)

        # Optional: Kaiming He initialization (for ReLU activations)
        # Set to False to use PyTorch's default initialization
        if use_kaiming_init:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming He initialization for ReLU networks."""
        # Initialize linear layers with Kaiming He
        torch.nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)
        
        # GCNConv layers - initialize their internal linear layers
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
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
        x = self.bn1(x) if not self.use_graph_norm else self.ln1(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 2
        identity = x  # Save current x
        x = self.conv2(x, edge_index)
        x = self.bn2(x) if not self.use_graph_norm else self.ln2(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 3
        identity = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x) if not self.use_graph_norm else self.ln3(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 4
        identity = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x) if not self.use_graph_norm else self.ln4(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 5
        identity = x
        x = self.conv5(x, edge_index)
        x = self.bn5(x) if not self.use_graph_norm else self.ln5(x, batch)
        x = F.relu(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!


        # Readout: aggregate node features to graph-level
        x = global_mean_pool(x, batch)
        
        # Final prediction
        x = self.linear(x)

        return x
