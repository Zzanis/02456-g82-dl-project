from enum import StrEnum
from typing import Type

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool, SAGEConv
from torch_geometric.nn.conv import GraphConv


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
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.05, use_layer_norm=False, use_residual=True, use_kaiming_init=False):
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
        x = self.bn1(x) if not self.use_layer_norm else self.ln1(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 2
        identity = x  # Save current x
        x = self.conv2(x, edge_index)
        x = self.bn2(x) if not self.use_layer_norm else self.ln2(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 3
        identity = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x) if not self.use_layer_norm else self.ln3(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 4
        identity = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x) if not self.use_layer_norm else self.ln4(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity  # RESIDUAL CONNECTION!

        # Layer 5
        identity = x
        x = self.conv5(x, edge_index)
        x = self.bn5(x) if not self.use_layer_norm else self.ln5(x, batch)
        x = F.relu(x)
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


#-------------- Improved Advanced GCN ---------------
class improved_advanced_GCN(torch.nn.Module):
    """
    Improved version of advanced_GCN that fixes architectural issues:
    - Reduced depth (4 layers instead of 6)
    - Increased width (default 128 channels)
    - Selective dropout application
    - Better head architecture
    - No custom initialization (uses PyTorch defaults with batch norm)
    """
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.2, use_layer_norm=False, use_residual=True):
        super(improved_advanced_GCN, self).__init__()
        
        # Reduced to 4 layers (better for molecular graphs)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        
        # Projection layer for first residual connection
        self.input_proj = torch.nn.Linear(num_node_features, hidden_channels, bias=False)
        
        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Optional: Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            from torch_geometric.nn import LayerNorm
            self.ln1 = LayerNorm(hidden_channels)
            self.ln2 = LayerNorm(hidden_channels)
            self.ln3 = LayerNorm(hidden_channels)
            self.ln4 = LayerNorm(hidden_channels)
        
        # Dropout - will be applied selectively
        self.dropout = torch.nn.Dropout(dropout)
        self.use_residual = use_residual
        
        # Improved head with intermediate layer
        self.pre_linear = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.final_bn = torch.nn.BatchNorm1d(hidden_channels // 2)
        self.linear = torch.nn.Linear(hidden_channels // 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Store original input for first residual connection
        x_input = self.input_proj(x) if self.use_residual else None

        # Layer 1: Conv -> Norm -> Activation -> (Selective Dropout)
        identity = x_input
        x = self.conv1(x, edge_index)
        x = self.bn1(x) if not self.use_layer_norm else self.ln1(x, batch)
        x = F.relu(x)
        # Skip dropout on first layer for better gradient flow
        if self.use_residual:
            x = x + identity

        # Layer 2: Full regularization
        identity = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x) if not self.use_layer_norm else self.ln2(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity

        # Layer 3: Full regularization
        identity = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x) if not self.use_layer_norm else self.ln3(x, batch)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity

        # Layer 4: Final conv layer (skip dropout before pooling)
        identity = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x) if not self.use_layer_norm else self.ln4(x, batch)
        x = F.relu(x)
        if self.use_residual:
            x = x + identity

        # Readout: aggregate node features to graph-level
        x = global_mean_pool(x, batch)
        
        # Improved prediction head
        x = self.pre_linear(x)
        x = self.final_bn(x)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout before final layer
        x = self.linear(x)

        return x


class NormLayerType(StrEnum):
    """Normalisation layer types with associated constructors"""

    BATCH = "batch"
    LAYER = "layer"
    NONE = "none"

    def __new__(cls, value: str) -> "NormLayerType":
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def new(self) -> Type[torch.nn.Module]:
        """Return the constructor for this layer type"""

        layer_map = {
            self.BATCH: BatchNorm,
            self.LAYER: LayerNorm,
            self.NONE: torch.nn.Identity,
        }
        return layer_map[self]


class GCN_MLP(torch.nn.Module):
    """GCN with a MLP at the end"""

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        dropout: float = 0,
        dropout_final: float = 0,
        use_residual: bool = False,
        norm_layer_type: NormLayerType = NormLayerType.NONE,
    ):
        super(GCN, self).__init__()
        self.p_dropout = dropout
        self.p_dropout_final = dropout_final
        self.use_residual = use_residual

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.norm1 = norm_layer_type.new(hidden_channels)
        self.norm2 = norm_layer_type.new(hidden_channels)
        self.norm3 = norm_layer_type.new(hidden_channels)
        self.norm4 = norm_layer_type.new(hidden_channels)
        self.norm5 = norm_layer_type.new(hidden_channels)
        self.residual1 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        if self.use_residual:
            x = x + self.residual1(x_res)
        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv5(x, edge_index)
        x = self.norm5(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv6(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p_dropout_final, training=self.training)
        x = self.mlp(x)

        return x


class GCN_GraphConv(torch.nn.Module):
    """GCN that uses GraphConv layers for convolution and a MLP at the end"""

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        dropout: float = 0,
        dropout_final: float = 0,
        use_residual: bool = False,
        norm_layer_type: NormLayerType = NormLayerType.NONE,
    ):
        super(GCN_GraphConv, self).__init__()
        self.p_dropout = dropout
        self.p_dropout_final = dropout_final
        self.use_residual = use_residual

        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.conv6 = GraphConv(hidden_channels, hidden_channels)
        self.norm1 = norm_layer_type.new(hidden_channels)
        self.norm2 = norm_layer_type.new(hidden_channels)
        self.norm3 = norm_layer_type.new(hidden_channels)
        self.norm4 = norm_layer_type.new(hidden_channels)
        self.norm5 = norm_layer_type.new(hidden_channels)
        self.residual1 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        if self.use_residual:
            x = x + self.residual1(x_res)
        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv5(x, edge_index)
        x = self.norm5(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv6(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p_dropout_final, training=self.training)
        x = self.mlp(x)

        return x
